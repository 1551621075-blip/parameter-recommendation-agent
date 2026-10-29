import hashlib
import json
from copy import deepcopy
from decimal import Decimal

import pytest
from pydantic import ValidationError

from parameter_agent.domain import (
    MAX_CONTEXT_AGE_MS,
    PARAMETER_NAMES,
    UNITS,
    ParameterValues,
    ProcessContext,
    canonical,
    check_candidate,
    digest,
    validate_context,
)
from parameter_agent.errors import DomainError
from parameter_agent.fixtures import demo_contexts

NOW = 1_750_010_000_000
ZERO = dict.fromkeys(PARAMETER_NAMES, "0")


def context(defect="short_shot", **updates):
    data = demo_contexts(NOW)[defect].model_dump(mode="json")
    data.update(updates)
    return ProcessContext.model_validate(data)


def delta(**updates):
    return {**ZERO, **updates}


def test_canonical_independent_oracle():
    value = {"z": [1, 2.25, True, None], "中文": "试验", "a": "0"}
    expected = '{"a":"0","z":[1,2.25,true,null],"中文":"试验"}'.encode()
    assert canonical(value) == expected
    assert digest(value) == hashlib.sha256(expected).hexdigest()
    assert digest(dict(reversed(list(value.items())))) == digest(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {1: "coercion"}, Decimal("1.5"), (1, 2), 2**64,
                                        "a" * 65537, "\ud800"])
def test_canonical_rejects_non_json_and_unbounded_values(value):
    with pytest.raises(DomainError):
        canonical(value)


def test_canonical_depth_and_cycles():
    value = []
    value.append(value)
    with pytest.raises(DomainError):
        canonical(value)


@pytest.mark.parametrize("value", [1, 1.0, True, None, "1e2", "NaN", "Infinity", "+1", " 1", "1 ", "01", ".5", "1.", "1\n", "0.0000000001", "9" * 25])
def test_strict_decimal_strings(value):
    with pytest.raises(ValidationError):
        ParameterValues(**{**ZERO, "temperature": value})


def test_strict_context_forbids_unknown_fields_coercion_nonfinite():
    base = context().model_dump(mode="json")
    for field, value in (("revision", True), ("revision", "1"), ("severity", float("nan")),
                         ("material_viscosity", float("inf")), ("sample_count", True),
                         ("measurement_ok", 1), ("synthetic", 1), ("synthetic", False),
                         ("context_id", "../escape"), ("ambient_c", True)):
        with pytest.raises(ValidationError):
            ProcessContext.model_validate({**base, field: value})
    with pytest.raises(ValidationError):
        ProcessContext.model_validate({**base, "hidden_override": True})
    with pytest.raises(ValidationError):
        ParameterValues(**{**ZERO, "extra": "1"})


def test_context_issues_and_exact_freshness():
    valid = context(observed_at_ms=NOW - MAX_CONTEXT_AGE_MS)
    assert validate_context(valid, NOW) == []
    assert "context_expired" in validate_context(context(observed_at_ms=NOW - MAX_CONTEXT_AGE_MS - 1), NOW)
    assert "context_from_future" in validate_context(context(observed_at_ms=NOW + 1), NOW)
    issues = validate_context(context(units={**UNITS, "pressure": "bar"}, process_version="other-v1",
                                      material_viscosity=None, ambient_c=None, sample_count=29, measurement_ok=False), NOW)
    assert set(issues) == {"units_mismatch", "unsupported_process_version", "missing_material_viscosity",
                           "missing_ambient_c", "insufficient_samples", "measurement_not_ok"}
    assert "ambient_c_out_of_range" in validate_context(context(ambient_c=46.0), NOW)
    assert "material_viscosity_out_of_range" in validate_context(context(material_viscosity=0.29), NOW)
    bad = context(current={"temperature": "240.000000001", "speed": "50.1", "pressure": "43", "cooling": "18"})
    assert set(validate_context(bad, NOW)) == {"temperature_out_of_range", "temperature_off_grid", "speed_off_grid"}


@pytest.mark.parametrize("value", [True, -1, 1.0, "1"])
def test_strict_now(value):
    with pytest.raises(DomainError):
        validate_context(context(), value)


def test_missing_is_not_imputed_and_no_mutation():
    ctx = demo_contexts(NOW)["missing"]
    before = deepcopy(ctx.model_dump(mode="json"))
    candidate = check_candidate(ctx, delta(temperature="2"), NOW)
    assert not candidate["eligible"]
    assert "missing_material_viscosity" in candidate["issues"]
    assert ctx.model_dump(mode="json") == before
    assert candidate["context_hash"] == digest(before)
    assert json.loads(json.dumps(candidate)) == candidate


@pytest.mark.parametrize("raw,expected", [("0.5", "0"), ("1.5", "2"), ("2.5", "2"), ("3.5", "4")])
def test_exact_half_even_temperature_target(raw, expected):
    # 204 is an even lattice coordinate; these are exact ties, not float noise.
    candidate = check_candidate(context(), delta(temperature=raw), NOW)
    assert candidate["delta"]["temperature"] == expected
    assert candidate["eligible"] == (expected != "0")
    assert candidate["raw_delta"]["temperature"] == raw


def test_half_even_is_target_lattice_not_delta_lattice():
    ctx = context(current={"temperature": "205", "speed": "55", "pressure": "43", "cooling": "18"})
    result = check_candidate(ctx, delta(temperature="0.5", speed="2.5"), NOW)
    assert result["target"]["temperature"] == "206"
    assert result["target"]["speed"] == "60"


@pytest.mark.parametrize("raw,issue", [
    (delta(temperature="6.000000001"), "raw_temperature_max_delta"),
    (delta(speed="10.000000001"), "raw_speed_max_delta"),
    (delta(pressure="4.000000001"), "raw_pressure_max_delta"),
    (delta(temperature="-0.000000001"), "raw_temperature_direction"),
    (delta(temperature="1", speed="0.000000001", pressure="1"), "raw_too_many_changes"),
    (delta(cooling="0.000000001"), "raw_cooling_direction"),
])
def test_invalid_raw_never_repaired_by_rounding(raw, issue):
    result = check_candidate(context(), raw, NOW)
    assert not result["eligible"]
    assert issue in result["issues"]
    assert result["raw_delta"] == raw


def test_target_range_exact_boundaries_not_clamped():
    ctx = context(current={"temperature": "234", "speed": "50", "pressure": "43", "cooling": "18"})
    assert check_candidate(ctx, delta(temperature="6"), NOW)["eligible"]
    result = check_candidate(ctx, delta(temperature="6.000000001"), NOW)
    assert result["target"]["temperature"] == "240"
    assert "raw_temperature_out_of_range" in result["issues"]
    ctx = context("flash", current={"temperature": "215", "speed": "25", "pressure": "34", "cooling": "19"})
    assert check_candidate(ctx, delta(speed="-5", pressure="-4"), NOW)["eligible"]
    assert "raw_speed_out_of_range" in check_candidate(ctx, delta(speed="-5.000000001"), NOW)["issues"]


def test_coupling_raw_and_rounded_separately():
    hot = context(current={"temperature": "229", "speed": "50", "pressure": "43", "cooling": "14"})
    candidate = check_candidate(hot, delta(temperature="0.6"), NOW)
    assert "quantized_temperature_cooling_coupling" in candidate["issues"]
    assert "raw_temperature_cooling_coupling" not in candidate["issues"]
    fast = context(current={"temperature": "204", "speed": "80", "pressure": "61", "cooling": "18"})
    candidate = check_candidate(fast, delta(speed="0.000000001"), NOW)
    assert "raw_speed_pressure_coupling" in candidate["issues"]
    assert "quantized_speed_pressure_coupling" not in candidate["issues"]
    assert not candidate["eligible"]


def test_all_defect_directions_independent_oracle():
    allowed = {"short_shot": {("temperature", 1), ("speed", 1), ("pressure", 1)},
               "flash": {("speed", -1), ("pressure", -1)},
               "warpage": {("temperature", -1), ("cooling", 1)}, "normal": set()}
    for defect, permitted in allowed.items():
        ctx = context(defect)
        for name in PARAMETER_NAMES:
            for sign in (-1, 1):
                step = 5 if name == "speed" else 1
                candidate = check_candidate(ctx, delta(**{name: str(sign * step)}), NOW)
                assert candidate["eligible"] == ((name, sign) in permitted)
                assert candidate["directions"][name] == ("increase" if sign > 0 else "decrease")
        assert check_candidate(ctx, ZERO, NOW)["issues"] == ["no_change"]


@pytest.mark.parametrize("raw", [{}, {"temperature": "1"}, {**ZERO, "extra": "1"}, {**ZERO, "speed": 5}, None])
def test_malformed_deltas_fail_explicitly(raw):
    with pytest.raises(DomainError):
        check_candidate(context(), raw, NOW)
