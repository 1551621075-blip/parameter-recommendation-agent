"""Original synthetic demonstrations and a toy process, not production or causal data.

The enumerating expert minimizes a known simulator expectation plus action cost.
BC learns those decisions from observations; its confidence is NOT defect probability.
Simulated observations are independent binomial draws, never guaranteed improvements.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from .domain import (
    BOUNDS,
    PARAMETER_NAMES,
    PROCESS_VERSION,
    UNITS,
    ParameterValues,
    ProcessContext,
    check_candidate,
    validate_context,
)
from .errors import DomainError

ZERO = {name: "0" for name in PARAMETER_NAMES}
_DEMO_TIME_MS = 1_750_000_000_000


def _context(*, identifier: str, now_ms: int, defect: str, current: dict[str, str],
             viscosity: float = 1.05, ambient: float = 23.0,
             severity: float = 0.65, machine: str = "synthetic-cell-demo") -> ProcessContext:
    return ProcessContext(
        context_id=identifier, plant_id="synthetic-plant", machine_id=machine,
        material_id="synthetic-resin-A", mold_id="synthetic-mold-A",
        process_version=PROCESS_VERSION, revision=1, observed_at_ms=now_ms,
        defect=defect, severity=severity, current=ParameterValues(**current),
        units=dict(UNITS), material_viscosity=viscosity, ambient_c=ambient,
        sample_count=60, measurement_ok=True, synthetic=True,
    )


def demo_contexts(now_ms: int) -> dict[str, ProcessContext]:
    # A zero timestamp is allowed; never fabricate a future observation.
    observed = max(0, now_ms - 1_000)
    contexts = {
        "short_shot": _context(identifier="synthetic-demo-short-shot", now_ms=observed,
                               machine="synthetic-cell-short_shot", defect="short_shot", current=dict(temperature="204", speed="50", pressure="43", cooling="18")),
        "flash": _context(identifier="synthetic-demo-flash", now_ms=observed,
                          machine="synthetic-cell-flash", defect="flash", current=dict(temperature="215", speed="75", pressure="60", cooling="19")),
        "warpage": _context(identifier="synthetic-demo-warpage", now_ms=observed,
                            machine="synthetic-cell-warpage", defect="warpage", current=dict(temperature="225", speed="55", pressure="48", cooling="14")),
        "normal": _context(identifier="synthetic-demo-normal", now_ms=observed,
                           machine="synthetic-cell-normal", defect="normal", severity=0.05,
                           current=dict(temperature="211", speed="55", pressure="47", cooling="20")),
    }
    missing = contexts["short_shot"].model_dump(mode="json")
    missing.update(context_id="synthetic-demo-missing", machine_id="synthetic-cell-missing", material_viscosity=None, ambient_c=None)
    contexts["missing"] = ProcessContext.model_validate(missing)
    ood = contexts["short_shot"].model_dump(mode="json")
    # Physically inside the example domain but outside demonstrated viscosity.
    ood.update(context_id="synthetic-demo-ood", machine_id="synthetic-cell-ood", material_viscosity=2.3)
    contexts["ood"] = ProcessContext.model_validate(ood)
    return contexts


def _expected_quality(context: ProcessContext, parameters: dict[str, str]) -> float:
    """Smooth, original toy penalty with viscosity/ambient-dependent process optima.

    Absolute setpoints matter. The observed severity scales, but does not label,
    the simulated defect probability. This is a deliberately misspecified small
    teaching process, not a fitted physical model of injection molding.
    """
    t, s, p, c = (float(parameters[name]) for name in PARAMETER_NAMES)
    v, a = context.material_viscosity, context.ambient_c
    if v is None or a is None:
        raise DomainError("missing_simulator_context", "simulator requires explicit context values", 422)
    if context.defect == "short_shot":
        deficits = (
            max(0.0, (213.0 + 12.0 * (v - 1.0) - 0.12 * (a - 23.0) - t) / 6.0),
            max(0.0, (65.0 - 15.0 * (v - 1.0) - s) / 10.0),
            max(0.0, (51.0 + 4.0 * (v - 1.0) - p) / 4.0),
        )
        burden = 0.65 * deficits[0] ** 2 + 0.65 * deficits[1] ** 2 + 0.65 * deficits[2] ** 2
    elif context.defect == "flash":
        burden = (0.8 * max(0.0, (s - (52.0 + 10.0 * (v - 1.0))) / 10.0) ** 2
                  + 0.8 * max(0.0, (p - (45.0 + 6.0 * (v - 1.0))) / 4.0) ** 2)
    elif context.defect == "warpage":
        burden = (0.9 * max(0.0, (t - (212.0 - 0.1 * (a - 23.0))) / 6.0) ** 2
                  + 0.9 * max(0.0, (22.0 + 0.15 * (a - 23.0) - c) / 3.0) ** 2)
    else:
        burden = (((t - 211.0) / 30.0) ** 2 + ((s - 55.0) / 50.0) ** 2
                  + ((p - 47.0) / 25.0) ** 2 + ((c - 20.0) / 15.0) ** 2) * 0.12
    # Saturating probability avoids values outside [0,1], without post-hoc labels.
    return float(0.015 + (0.42 + 0.28 * context.severity) * (1.0 - math.exp(-0.18 * burden)))


def simulate_quality(context: ProcessContext, parameters: dict[str, str], seed: int,
                     sample_count: int = 60) -> dict:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise DomainError("invalid_seed", "seed must be a bounded nonnegative integer", 422)
    if type(sample_count) is not int or not 1 <= sample_count <= 1_000_000:
        raise DomainError("invalid_sample_count", "sample_count must be in 1..1000000", 422)
    issues = validate_context(context, context.observed_at_ms)
    if issues:
        raise DomainError("invalid_simulator_context", ", ".join(issues), 422)
    try:
        values = ParameterValues.model_validate(parameters)
        proposed = context.model_dump(mode="json")
        proposed["current"] = values.model_dump(mode="json")
        issues = validate_context(ProcessContext.model_validate(proposed), context.observed_at_ms)
    except (ValueError, TypeError) as exc:
        raise DomainError("invalid_simulator_parameters", "invalid parameter strings", 422) from exc
    if issues:
        raise DomainError("invalid_simulator_parameters", ", ".join(issues), 422)
    expectation = _expected_quality(context, values.model_dump(mode="json"))
    count = int(np.random.default_rng(seed).binomial(sample_count, expectation))
    return {"synthetic": True, "sample_count": sample_count, "defect_count": count,
            "defect_rate": count / sample_count, "expected_defect_rate": expectation}


def _expert_delta(context: ProcessContext) -> dict[str, str]:
    """Enumerate finite actions, independently gate, minimize expectation + cost.

    Cost is 0.004 per changed axis + 0.002 per squared normalized step. Zero is
    always considered; deterministic lexicographic tie-breaking is not learning.
    No calibration/test records, random future observations or BC weights enter.
    """
    levels = {
        "short_shot": ((0, 2, 4, 6), (0, 5, 10), (0, 1, 2, 3, 4), (0,)),
        "flash": ((0,), (0, -5, -10), (0, -1, -2, -3, -4), (0,)),
        "warpage": ((0, -2, -4, -6), (0,), (0,), (0, 1, 2, 3)),
        "normal": ((0,), (0,), (0,), (0,)),
    }[context.defect]
    best = dict(ZERO)
    best_objective = _expected_quality(context, context.current.model_dump(mode="json"))
    for action in itertools.product(*levels):
        if not 1 <= sum(value != 0 for value in action) <= 2:
            continue
        raw = dict(zip(PARAMETER_NAMES, map(str, action), strict=True))
        candidate = check_candidate(context, raw, context.observed_at_ms)
        if not candidate["eligible"]:
            continue
        cost = 0.004 * sum(value != 0 for value in action)
        cost += 0.002 * sum((value / float(BOUNDS[name][3])) ** 2
                            for name, value in zip(PARAMETER_NAMES, action, strict=True))
        objective = _expected_quality(context, candidate["target"]) + cost
        if objective < best_objective - 1e-12:
            best_objective, best = objective, candidate["delta"]
    return best


def generate_demonstrations(seed: int = 41723, groups: int = 120,
                            per_group: int = 12) -> list[dict]:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise DomainError("invalid_seed", "seed must be a bounded nonnegative integer", 422)
    if type(groups) is not int or type(per_group) is not int or not 10 <= groups <= 500 or not 4 <= per_group <= 100 or groups * per_group > 10_000:
        raise DomainError("invalid_demonstration_size", "use 10..500 groups, 4..100 rows/group, at most 10000 rows", 422)
    rng = np.random.default_rng(seed)
    records = []
    for group in range(groups):
        # Each group is a distinct synthetic device/batch lineage. Replicates
        # share viscosity/ambient and device, and MUST remain in one partition.
        group_id = f"synthetic-batch-{group:04d}"
        viscosity = float(rng.uniform(0.75, 1.35))
        ambient = float(rng.uniform(18.0, 29.0))
        for index in range(per_group):
            defect = ("short_shot", "flash", "warpage", "normal")[index % 4]
            if defect == "short_shot":
                values = (rng.integers(199, 219), rng.choice((40, 45, 50, 55, 60, 65, 70)), rng.integers(39, 55), rng.integers(16, 25))
            elif defect == "flash":
                values = (rng.integers(205, 224), rng.choice((50, 55, 60, 65, 70, 75, 80)), rng.integers(44, 65), rng.integers(16, 25))
            elif defect == "warpage":
                values = (rng.integers(211, 230), rng.choice((45, 50, 55, 60, 65)), rng.integers(40, 56), rng.integers(11, 23))
            else:
                values = (rng.integers(205, 219), rng.choice((45, 50, 55, 60, 65)), rng.integers(42, 54), rng.integers(16, 25))
            context = _context(
                identifier=f"synthetic-context-{group:04d}-{index:03d}",
                now_ms=_DEMO_TIME_MS + group * 10_000 + index,
                machine=f"synthetic-cell-{group:04d}", defect=defect,
                current=dict(zip(PARAMETER_NAMES, (str(int(value)) for value in values), strict=True)),
                viscosity=viscosity, ambient=ambient,
                severity=float(rng.uniform(0.35, 0.85)) if defect != "normal" else float(rng.uniform(0.0, 0.12)),
            )
            records.append({"group_id": group_id, "context": context.model_dump(mode="json"),
                            "delta": _expert_delta(context)})
    return records
