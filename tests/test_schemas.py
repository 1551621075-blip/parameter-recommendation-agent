"""Strict public request boundaries, using only original synthetic input."""

from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError

from parameter_agent.fixtures import demo_contexts
from parameter_agent.schemas import (
    ActionRequest,
    ContextRequest,
    CreateTaskRequest,
    ExecuteRequest,
    ReconcileRequest,
    ReviewRequest,
    SupplementRequest,
)

NOW = 1_800_000_000_000
HASH = "a" * 64
REQUESTS = (
    (ActionRequest, {"action_id": "r5-action"}),
    (ContextRequest, {"action_id": "r5-import",
                      "context": demo_contexts(NOW)["short_shot"].model_dump(mode="json")}),
    (CreateTaskRequest, {"action_id": "r5-create", "context_id": "synthetic-context"}),
    (SupplementRequest, {"action_id": "r5-supplement",
                         "fields": {"material_viscosity": 1.05, "ambient_c": 23.0},
                         "note": "Explicit synthetic measurements supplied by the operator."}),
    (ReviewRequest, {"action_id": "r5-review", "recommendation_hash": HASH,
                    "decision": "approved", "note": "Synthetic candidate reviewed."}),
    (ExecuteRequest, {"action_id": "r5-execute", "recommendation_hash": HASH, "confirmed": True}),
    (ReconcileRequest, {"action_id": "r5-reconcile", "call_id": "call-r5", "no_result": True,
                       "tokens": None, "note": "No result was recovered; usage is unknown."}),
)


@pytest.mark.parametrize("schema,body", REQUESTS)
def test_valid_requests_round_trip_strict_json(schema, body):
    request = schema.model_validate(copy.deepcopy(body))
    assert request.model_dump(mode="json") == body
    assert schema.model_validate_json(json.dumps(body)).model_dump(mode="json") == body


@pytest.mark.parametrize("schema,body", REQUESTS)
@pytest.mark.parametrize("field", ["owner", "role", "token", "endpoint", "path", "seed", "fault"])
def test_server_owned_fields_cannot_be_smuggled_into_requests(schema, body, field):
    with pytest.raises(ValidationError):
        schema.model_validate({**copy.deepcopy(body), field: "untrusted"})


@pytest.mark.parametrize("schema,body", REQUESTS)
@pytest.mark.parametrize("identifier", [None, True, 12, "", "a" * 81, "../file", "a/b", " a", "a\n"])
def test_action_ids_are_strict_bounded_identifiers(schema, body, identifier):
    with pytest.raises(ValidationError):
        schema.model_validate({**copy.deepcopy(body), "action_id": identifier})


@pytest.mark.parametrize("schema,body", [REQUESTS[4], REQUESTS[5]])
@pytest.mark.parametrize("value", [None, True, 123, "A" * 64, "a" * 63, "a" * 65, "g" * 64])
def test_recommendation_hash_is_exact_lowercase_sha256(schema, body, value):
    with pytest.raises(ValidationError):
        schema.model_validate({**body, "recommendation_hash": value})


@pytest.mark.parametrize("schema,body,field", [
    (*REQUESTS[5], "confirmed"), (*REQUESTS[6], "no_result"),
])
@pytest.mark.parametrize("value", [False, 0, 1, 1.0, "true", "True", "1", None, [], {}])
def test_confirmation_and_no_result_require_literal_boolean_true(schema, body, field, value):
    with pytest.raises(ValidationError):
        schema.model_validate({**body, field: value})


@pytest.mark.parametrize("tokens", [True, False, -1, 100001, 1.0, "1", float("nan"), float("inf")])
def test_reconciliation_rejects_coerced_or_unbounded_usage(tokens):
    with pytest.raises(ValidationError):
        ReconcileRequest.model_validate({**REQUESTS[6][1], "tokens": tokens})


@pytest.mark.parametrize("tokens", [None, 0, 1, 100000])
def test_reconciliation_keeps_unknown_distinct_from_zero(tokens):
    request = ReconcileRequest.model_validate({**REQUESTS[6][1], "tokens": tokens})
    assert request.tokens == tokens
    assert type(request.tokens) is type(tokens)


@pytest.mark.parametrize("schema,body", [REQUESTS[3], REQUESTS[4], REQUESTS[6]])
@pytest.mark.parametrize("note", ["", " ", "\t\n", "　", "n" * 2001, None, 42, True])
def test_human_notes_are_bounded_nonblank_strings(schema, body, note):
    with pytest.raises(ValidationError):
        schema.model_validate({**body, "note": note})


@pytest.mark.parametrize("fields", [
    {}, {"current": 211.0}, {"observed_at_ms": 1.0}, {"revision": 2.0},
    {"material_viscosity": 1.0, "extra": 1.0},
    {"material_viscosity": 0.29}, {"material_viscosity": 2.51},
    {"ambient_c": 4.99}, {"ambient_c": 45.01},
    {"ambient_c": True}, {"ambient_c": "23"}, {"ambient_c": None},
    {"ambient_c": float("nan")}, {"ambient_c": float("inf")},
])
def test_supplement_only_accepts_nonempty_missing_measurement_fields(fields):
    with pytest.raises(ValidationError):
        SupplementRequest.model_validate({**REQUESTS[3][1], "fields": fields})


@pytest.mark.parametrize("fields", [
    {"material_viscosity": 0.3}, {"material_viscosity": 2.5},
    {"ambient_c": 5.0}, {"ambient_c": 45.0},
])
def test_supplement_measurement_bounds_are_inclusive(fields):
    request = SupplementRequest.model_validate({**REQUESTS[3][1], "fields": fields})
    assert request.fields == fields


@pytest.mark.parametrize("decision", ["executed", "pending_review", "APPROVED", True, None])
def test_review_cannot_select_execution_or_invent_a_state(decision):
    with pytest.raises(ValidationError):
        ReviewRequest.model_validate({**REQUESTS[4][1], "decision": decision})


@pytest.mark.parametrize("path,value", [
    (("synthetic",), 1), (("synthetic",), False), (("measurement_ok",), "true"),
    (("revision",), True), (("sample_count",), "60"), (("severity",), float("nan")),
    (("severity",), float("inf")), (("current", "temperature"), 204.0),
    (("current", "temperature"), True), (("current", "temperature"), "2.04e2"),
    (("current", "pressure"), "NaN"), (("current", "cooling"), "18; DROP TABLE calls"),
    (("current", "extra"), "1"), (("owner",), "another-user"),
])
def test_context_request_retains_nested_domain_validation(path, value):
    body = copy.deepcopy(REQUESTS[1][1])
    target = body["context"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        ContextRequest.model_validate(body)


def test_missing_measurements_remain_explicit_null_not_imputed():
    body = {"action_id": "r5-missing",
            "context": demo_contexts(NOW)["missing"].model_dump(mode="json")}
    request = ContextRequest.model_validate(body)
    assert request.context.material_viscosity is None
    assert request.context.ambient_c is None


def test_untrusted_model_like_text_remains_text_not_an_executable_field():
    note = '<script>alert("untrusted synthetic note")</script>'
    request = ReviewRequest.model_validate({**REQUESTS[4][1], "note": note})
    assert request.note == note
    assert set(request.model_dump()) == {"action_id", "recommendation_hash", "decision", "note"}
