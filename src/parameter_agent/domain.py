"""Original synthetic molding domain; these limits are examples, not plant rules."""
from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from .errors import DomainError

PARAMETER_NAMES = ("temperature", "speed", "pressure", "cooling")
UNITS = {"temperature": "degC", "speed": "mm/s", "pressure": "MPa", "cooling": "s"}
PROCESS_VERSION = "synthetic-molding-v1"
CONSTRAINT_VERSION = "synthetic-bop-v1"
MAX_CONTEXT_AGE_MS = 24 * 60 * 60 * 1000
# low, high, grid, maximum absolute change. Decimal is never serialized publicly.
BOUNDS = {
    "temperature": tuple(map(Decimal, ("180", "240", "1", "6"))),
    "speed": tuple(map(Decimal, ("20", "100", "5", "10"))),
    "pressure": tuple(map(Decimal, ("30", "70", "1", "4"))),
    "cooling": tuple(map(Decimal, ("8", "30", "1", "3"))),
}
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$", min_length=1, max_length=80)]
_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]{1,9})?\Z")


def canonical(value) -> bytes:
    """Bounded canonical JSON, without coercion of keys, numbers or custom objects."""
    remaining = 150_000

    def visit(item, depth):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 32:
            raise ValueError("JSON structure exceeds limits")
        if item is None or type(item) is bool:
            return
        if type(item) is str:
            if len(item) > 65_536:
                raise ValueError("JSON string exceeds limit")
        elif type(item) is int:
            if abs(item) > 2**63 - 1:
                raise ValueError("JSON integer exceeds limit")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("nonfinite JSON number")
        elif type(item) is list:
            if len(item) > 50_000:
                raise ValueError("JSON array exceeds limit")
            for child in item:
                visit(child, depth + 1)
        elif type(item) is dict:
            if len(item) > 50_000:
                raise ValueError("JSON object exceeds limit")
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("JSON keys must be strings")
                visit(key, depth + 1)
                visit(child, depth + 1)
        else:
            raise ValueError("not a JSON value")

    try:
        visit(value, 0)
        result = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                            separators=(",", ":")).encode("utf-8", errors="strict")
        if len(result) > 4 * 1024 * 1024:
            raise ValueError("JSON document exceeds limit")
        return result
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise DomainError("invalid_json", str(exc), 422) from exc


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ParameterValues(StrictModel):
    temperature: str
    speed: str
    pressure: str
    cooling: str

    @field_validator("temperature", "speed", "pressure", "cooling", mode="before")
    @classmethod
    def decimal_string(cls, value):
        if type(value) is not str or len(value) > 24 or not _DECIMAL.fullmatch(value):
            raise ValueError("requires a bounded plain decimal string (at most 9 fractional digits)")
        return value


class ProcessContext(StrictModel):
    context_id: Identifier
    plant_id: Identifier
    machine_id: Identifier
    material_id: Identifier
    mold_id: Identifier
    process_version: Identifier
    revision: int = Field(ge=1)
    observed_at_ms: int = Field(ge=0, le=2**63 - 1)
    defect: Literal["short_shot", "flash", "warpage", "normal"]
    severity: float = Field(ge=0, le=1)
    current: ParameterValues
    units: dict[str, str]
    material_viscosity: float | None = None
    ambient_c: float | None = None
    sample_count: int = Field(ge=0, le=1_000_000)
    measurement_ok: bool
    synthetic: Literal[True]

    @field_validator("synthetic", mode="before")
    @classmethod
    def synthetic_only(cls, value):
        if value is not True:
            raise ValueError("only explicitly synthetic contexts are accepted")
        return value


def decimal_text(value: Decimal) -> str:
    if not value:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _coupling(target: dict[str, Decimal]) -> list[str]:
    issues = []
    if target["temperature"] >= Decimal("230") and target["cooling"] < Decimal("15"):
        issues.append("temperature_cooling_coupling")
    if target["speed"] > Decimal("80") and target["pressure"] > Decimal("60"):
        issues.append("speed_pressure_coupling")
    return issues


def validate_context(context: ProcessContext, now_ms: int) -> list[str]:
    if type(now_ms) is not int or now_ms < 0:
        raise DomainError("invalid_time", "now_ms must be a nonnegative integer", 422)
    # Revalidation also protects against Pydantic model_copy/model_construct bypasses.
    try:
        context = ProcessContext.model_validate(context.model_dump(mode="json"))
    except (ValueError, TypeError, AttributeError) as exc:
        raise DomainError("invalid_context", "context schema is invalid", 422) from exc
    issues = []
    if context.process_version != PROCESS_VERSION:
        issues.append("unsupported_process_version")
    if context.units != UNITS:
        issues.append("units_mismatch")
    if context.material_viscosity is None:
        issues.append("missing_material_viscosity")
    elif not 0.3 <= context.material_viscosity <= 2.5:
        issues.append("material_viscosity_out_of_range")
    if context.ambient_c is None:
        issues.append("missing_ambient_c")
    elif not 5 <= context.ambient_c <= 45:
        issues.append("ambient_c_out_of_range")
    if not context.measurement_ok:
        issues.append("measurement_not_ok")
    if context.sample_count < 30:
        issues.append("insufficient_samples")
    if context.observed_at_ms > now_ms:
        issues.append("context_from_future")
    elif now_ms - context.observed_at_ms > MAX_CONTEXT_AGE_MS:
        issues.append("context_expired")
    values = {name: Decimal(getattr(context.current, name)) for name in PARAMETER_NAMES}
    with localcontext() as arithmetic:
        arithmetic.prec = 50
        for name, value in values.items():
            low, high, grid, _ = BOUNDS[name]
            if not low <= value <= high:
                issues.append(f"{name}_out_of_range")
            if (value - low) % grid:
                issues.append(f"{name}_off_grid")
        issues.extend(_coupling(values))
    return issues


def _action_issues(context, delta, target, phase):
    issues = []
    changed = sum(value != 0 for value in delta.values())
    if changed > 2:
        issues.append(f"{phase}_too_many_changes")
    allowed = {
        "short_shot": {"temperature": 1, "speed": 1, "pressure": 1},
        "flash": {"speed": -1, "pressure": -1},
        "warpage": {"temperature": -1, "cooling": 1},
        "normal": {},
    }[context.defect]
    for name in PARAMETER_NAMES:
        low, high, _, max_delta = BOUNDS[name]
        if abs(delta[name]) > max_delta:
            issues.append(f"{phase}_{name}_max_delta")
        if not low <= target[name] <= high:
            issues.append(f"{phase}_{name}_out_of_range")
        if delta[name] and (name not in allowed or delta[name] * allowed[name] < 0):
            issues.append(f"{phase}_{name}_direction")
    issues.extend(f"{phase}_{issue}" for issue in _coupling(target))
    return issues


def check_candidate(context: ProcessContext, raw_delta: dict[str, str], now_ms: int) -> dict:
    issues = validate_context(context, now_ms)
    try:
        raw = ParameterValues.model_validate(raw_delta).model_dump(mode="json")
    except (ValueError, TypeError) as exc:
        raise DomainError("invalid_delta", "delta must contain exactly four plain decimal strings", 422) from exc
    with localcontext() as arithmetic:
        arithmetic.prec = 50
        current = {name: Decimal(getattr(context.current, name)) for name in PARAMETER_NAMES}
        delta = {name: Decimal(raw[name]) for name in PARAMETER_NAMES}
        target = {name: current[name] + delta[name] for name in PARAMETER_NAMES}
        issues.extend(_action_issues(context, delta, target, "raw"))
        # Round TARGETS to the nearest legal lattice, not to a bounding interval.
        # Raw violations remain fatal even if rounding happens to remove them.
        quantized = {}
        for name in PARAMETER_NAMES:
            low, _, grid, _ = BOUNDS[name]
            quantized[name] = low + ((target[name] - low) / grid).quantize(
                Decimal("1"), rounding=ROUND_HALF_EVEN) * grid
        rounded_delta = {name: quantized[name] - current[name] for name in PARAMETER_NAMES}
        issues.extend(_action_issues(context, rounded_delta, quantized, "quantized"))
        if not any(rounded_delta.values()):
            issues.append("no_change")
    return {
        "eligible": not issues,
        "issues": issues,
        "raw_delta": raw,
        "delta": {name: decimal_text(rounded_delta[name]) for name in PARAMETER_NAMES},
        "target": {name: decimal_text(quantized[name]) for name in PARAMETER_NAMES},
        "directions": {name: ("increase" if value > 0 else "decrease" if value < 0 else "hold")
                       for name, value in rounded_delta.items()},
        "context_hash": digest(context.model_dump(mode="json")),
        "constraint_version": CONSTRAINT_VERSION,
    }
