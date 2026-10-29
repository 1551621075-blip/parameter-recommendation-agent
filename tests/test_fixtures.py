import itertools
import math
from decimal import Decimal

import numpy as np
import pytest

from parameter_agent.domain import PARAMETER_NAMES, ProcessContext, validate_context
from parameter_agent.errors import DomainError
from parameter_agent.fixtures import demo_contexts, generate_demonstrations, simulate_quality

NOW = 1_750_010_000_000


def test_original_demo_contexts():
    demos = demo_contexts(NOW)
    assert set(demos) == {"short_shot", "flash", "warpage", "normal", "missing", "ood"}
    assert len({item.machine_id for item in demos.values()}) == 6
    for name, context in demos.items():
        assert context.context_id.startswith("synthetic-")
        assert context.plant_id == "synthetic-plant"
        assert context.machine_id == "synthetic-cell-" + name
        assert context.process_version == "synthetic-molding-v1"
        assert context.observed_at_ms <= NOW
        assert context.measurement_ok is True and context.sample_count >= 30
        assert not validate_context(context, NOW) if name != "missing" else validate_context(context, NOW)
    assert all(item.observed_at_ms == 0 for item in demo_contexts(0).values())


def test_simulator_independent_arithmetic_and_binomial_oracles():
    ctx = demo_contexts(NOW)["short_shot"]
    raw = ctx.current.model_dump(mode="json")
    # Independently substitute known context constants into documented toy math.
    deficits = [(213 + 12 * 0.05 - 204) / 6, (65 - 15 * 0.05 - 50) / 10, (51 + 4 * 0.05 - 43) / 4]
    burden = 0.65 * sum(item * item for item in deficits)
    probability = 0.015 + (0.42 + 0.28 * 0.65) * (1 - math.exp(-0.18 * burden))
    result = simulate_quality(ctx, raw, seed=17, sample_count=123)
    assert result["expected_defect_rate"] == pytest.approx(probability, abs=1e-14)
    assert result["defect_count"] == int(np.random.default_rng(17).binomial(123, probability))
    assert result["defect_rate"] == result["defect_count"] / 123
    assert result["synthetic"] is True
    assert result == simulate_quality(ctx, raw, seed=17, sample_count=123)


def test_simulator_not_canned_improvement():
    ctx = demo_contexts(NOW)["short_shot"]
    current = ctx.current.model_dump(mode="json")
    baseline = simulate_quality(ctx, current, 1)["expected_defect_rate"]
    better = simulate_quality(ctx, {**current, "temperature": "210", "pressure": "47"}, 1)["expected_defect_rate"]
    worse = simulate_quality(ctx, {**current, "temperature": "198", "pressure": "39"}, 1)["expected_defect_rate"]
    assert better < baseline < worse
    changed = ctx.model_dump(mode="json")
    changed["material_viscosity"] = 1.3
    assert simulate_quality(ProcessContext.model_validate(changed), current, 1)["expected_defect_rate"] != baseline
    observations = {simulate_quality(ctx, current, seed)["defect_count"] for seed in range(30)}
    assert len(observations) > 5


@pytest.mark.parametrize("values", [{"temperature": "241", "speed": "50", "pressure": "43", "cooling": "18"},
                                      {"temperature": "204", "speed": "50.1", "pressure": "43", "cooling": "18"}])
def test_simulator_does_not_accept_invalid_setpoints(values):
    with pytest.raises(DomainError):
        simulate_quality(demo_contexts(NOW)["short_shot"], values, 1)


@pytest.mark.parametrize("sample_count", [True, 0, -1, 1.0, 1_000_001])
def test_simulator_sample_bounds(sample_count):
    ctx = demo_contexts(NOW)["short_shot"]
    with pytest.raises(DomainError):
        simulate_quality(ctx, ctx.current.model_dump(mode="json"), 1, sample_count)


def test_simulator_requires_explicit_missing_values():
    ctx = demo_contexts(NOW)["missing"]
    with pytest.raises(DomainError):
        simulate_quality(ctx, ctx.current.model_dump(mode="json"), 1)


@pytest.fixture(scope="module")
def small_records():
    return generate_demonstrations(groups=10, per_group=8)


def test_demonstrations_determinism_lineage_and_independent_safety(small_records):
    assert small_records == generate_demonstrations(groups=10, per_group=8)
    assert small_records != generate_demonstrations(seed=8, groups=10, per_group=8)
    devices, groups = {}, {}
    allowed = {"short_shot": {"temperature": 1, "speed": 1, "pressure": 1},
               "flash": {"speed": -1, "pressure": -1},
               "warpage": {"temperature": -1, "cooling": 1}, "normal": {}}
    limits = ((180, 240, 1, 6), (20, 100, 5, 10), (30, 70, 1, 4), (8, 30, 1, 3))
    for row in small_records:
        ctx = row["context"]
        assert ctx["plant_id"] == "synthetic-plant"
        groups.setdefault(row["group_id"], []).append(row)
        if ctx["machine_id"] in devices:
            assert devices[ctx["machine_id"]] == row["group_id"]
        devices[ctx["machine_id"]] = row["group_id"]
        changes = [Decimal(row["delta"][name]) for name in PARAMETER_NAMES]
        assert sum(value != 0 for value in changes) <= 2
        targets = []
        for name, change, (low, high, grid, maximum) in zip(PARAMETER_NAMES, changes, limits, strict=True):
            target = Decimal(ctx["current"][name]) + change
            assert Decimal(low) <= target <= Decimal(high)
            assert abs(change) <= Decimal(maximum)
            assert (target - low) % grid == 0
            if change:
                assert name in allowed[ctx["defect"]]
                assert change * allowed[ctx["defect"]][name] > 0
            targets.append(target)
        assert targets[0] < 230 or targets[3] >= 15
        assert targets[1] <= 80 or targets[2] <= 60
    assert len(groups) == 10
    assert {len(rows) for rows in groups.values()} == {8}


def test_expert_enumeration_objective_oracle(small_records):
    # Independently enumerate flash actions and calculate the objective using the
    # public simulator. Do not call _expert_delta or check_candidate as oracle.
    row = next(item for item in small_records if item["context"]["defect"] == "flash")
    ctx = ProcessContext.model_validate(row["context"])
    current = ctx.current.model_dump(mode="json")
    outcomes = []
    for speed, pressure in itertools.product((0, -5, -10), (0, -1, -2, -3, -4)):
        target = {**current, "speed": str(int(current["speed"]) + speed), "pressure": str(int(current["pressure"]) + pressure)}
        if int(target["speed"]) < 20 or int(target["pressure"]) < 30:
            continue
        expected = simulate_quality(ctx, target, 0)["expected_defect_rate"]
        cost = 0.004 * ((speed != 0) + (pressure != 0)) + 0.002 * ((speed / 10) ** 2 + (pressure / 4) ** 2)
        outcomes.append((expected + cost, speed, pressure))
    chosen = row["delta"]
    actual = next(item[0] for item in outcomes if item[1:] == (int(chosen["speed"]), int(chosen["pressure"])))
    assert actual == pytest.approx(min(item[0] for item in outcomes), abs=1e-14)


@pytest.mark.parametrize("kwargs", [{"groups": True}, {"groups": 9}, {"groups": 501}, {"per_group": 3},
                                     {"per_group": 101}, {"groups": 500, "per_group": 100}, {"seed": -1}])
def test_generation_bounded(kwargs):
    with pytest.raises(DomainError):
        generate_demonstrations(**kwargs)
