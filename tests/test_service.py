"""Service boundary tests use real domain, trained BC, and SQLite/artifact storage."""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier

import pytest

from parameter_agent import bc
from parameter_agent.domain import (
    MAX_CONTEXT_AGE_MS,
    ProcessContext,
    canonical,
    check_candidate,
    digest,
    validate_context,
)
from parameter_agent.errors import DomainError
from parameter_agent.fixtures import demo_contexts, generate_demonstrations
from parameter_agent.service import EXPERIENCE_VERSION, SOURCE_KEYS, RecommendationService
from parameter_agent.storage import Store

NOW = 1_750_000_000_000
PLANT = demo_contexts(NOW)["flash"].plant_id
SECOND_PLANT = "synthetic-plant-beta"


@dataclass
class Clock:
    now: int = NOW

    def __call__(self):
        return self.now


@pytest.fixture(scope="module")
def model():
    # Train actual NumPy weights; no hand-written prediction or mock core gates.
    return bc.train_model(generate_demonstrations(groups=20, per_group=12))["model"]


@pytest.fixture
def env(tmp_path):
    store = Store((tmp_path / "local-data").resolve())
    principal = store.create_user("owner", "a" * 32, "reviewer", [PLANT, SECOND_PLANT])
    clock = Clock()
    return store, principal, RecommendationService(store, clock=clock), clock


def context_for(name="flash", **changes):
    data = demo_contexts(NOW)[name].model_dump(mode="json")
    data.update(changes)
    return ProcessContext.model_validate(data)


def import_one(env, context=None, *, action="import-1"):
    _, principal, service, _ = env
    context = context or context_for()
    return service.import_context(principal, context, action)


def freeze(env, model, *, task_id="task-one", context=None):
    store, principal, service, clock = env
    obj = import_one(env, context)
    context_id = obj["object_id"]
    service.register_model(principal, model, obj["plant_id"], "register-1")
    snapshot = service.snapshot(principal, context_id)
    source = {key: snapshot[key] for key in SOURCE_KEYS}
    parsed = ProcessContext.model_validate(snapshot["context"])
    prediction = bc.predict(model, parsed, clock())
    assert prediction["status"] == "ok", prediction
    candidate = check_candidate(parsed, prediction["raw_delta"], clock())
    assert candidate["eligible"]
    candidate.update(task_id=task_id, context_id=context_id, model_hash=model["model_hash"])
    candidate["candidate_hash"] = digest(candidate)
    results = {
        "get_process_context": {"context": snapshot["context"],
                                "issues": validate_context(parsed, clock())},
        "retrieve_experience": service.experience(principal, context_id),
        "predict_bc": prediction,
        "check_constraints": {"candidate": candidate},
    }
    evidence = []
    for index, (tool, payload) in enumerate(results.items()):
        item = {"evidence_id": f"ev-{index}", "task_id": task_id, "tool": tool,
                "context_hash": source["context_hash"], "payload": payload}
        item["content_hash"] = digest(item)
        evidence.append(item)
    with store.transaction() as conn:
        store.create_object(
            conn, kind="task", object_id=task_id, principal=principal,
            plant_id=obj["plant_id"], status="running", payload={
                "status": "running", "context_id": context_id, "source_snapshot": source,
                "prediction": prediction, "candidate": candidate, "evidence": evidence,
            },
        )
    return {"task_id": task_id, "context_id": context_id, "expected_snapshot": source,
            "candidate": candidate, "evidence": evidence,
            "explanation": "Original synthetic comparison; unreviewed explanatory draft."}


def submit(env, frozen):
    store, principal, service, _ = env
    with store.transaction() as conn:
        return service.submit_recommendation(conn, principal, **frozen)


def update(env, kind, object_id, change, *, status=None):
    store, principal, _, _ = env
    with store.transaction() as conn:
        obj = store.get_object(kind, object_id, principal, conn=conn)
        payload = copy.deepcopy(obj["payload"])
        change(payload)
        return store.update_object(
            conn, kind=kind, object_id=object_id, principal=principal,
            expected_version=obj["version"], status=status or obj["status"], payload=payload,
        )


def supplement(env, object_id, expected_hash, fields, note="Explicit synthetic measurement"):
    store, principal, service, _ = env
    with store.transaction() as conn:
        return service.supplement_context(conn, principal, object_id, expected_hash=expected_hash,
                                          fields=fields, note=note)


def test_import_is_atomic_append_only_and_idempotent(env):
    store, principal, service, _ = env
    context = context_for()
    first = import_one(env, context)
    assert service.import_context(principal, context, "import-1") == first
    assert first["version"] == 1
    assert first["payload"] == context.model_dump(mode="json")
    device = store.get_object("device", context.machine_id, principal)
    assert device["payload"]["current"] == first["payload"]["current"]
    assert device["payload"]["context_hash"] == first["content_hash"]
    assert device["payload"]["revision"] == 1
    assert len(store.audit_events(principal)) == 2
    with pytest.raises(DomainError, match="append-only"):
        service.import_context(principal, context, "new-action")
    changed = context_for(severity=0.9)
    with pytest.raises(DomainError) as failure:
        service.import_context(principal, changed, "import-1")
    assert failure.value.code == "idempotency_conflict"
    assert service.get_context(principal, context.context_id) == first


def test_import_does_not_reset_existing_device(env):
    store, principal, service, _ = env
    original = import_one(env)
    other = context_for("warpage", machine_id=original["payload"]["machine_id"])
    with pytest.raises(DomainError) as failure:
        service.import_context(principal, other, "other-context")
    assert failure.value.code == "device_conflict"
    assert len(store.list_objects("context", principal)) == 1
    assert store.get_object("device", other.machine_id, principal)["version"] == 1


def test_context_and_model_scope_is_owner_and_plant(env, model):
    store, principal, service, _ = env
    obj = import_one(env)
    service.register_model(principal, model, PLANT, "model-owner")
    other = store.create_user("other", "b" * 32, "reviewer", [PLANT])
    with pytest.raises(DomainError) as failure:
        service.get_context(other, obj["object_id"])
    assert failure.value.status == 404
    assert service.get_model(other, PLANT) is None
    # Registry IDs bind owner+plant even though storage object IDs are global.
    second = service.register_model(other, model, PLANT, "model-other")
    assert second["owner"] == "other"
    assert second["object_id"] == "active-model-" + digest({"owner": "other", "plant_id": PLANT})
    assert service.get_model(other, PLANT) == second
    original_registry = service.get_model(principal, PLANT)
    assert original_registry["object_id"] != second["object_id"]
    assert original_registry["version"] == 1
    other_context = context_for(context_id="other-context", machine_id="other-cell")
    assert service.import_context(other, other_context, "import-other")["owner"] == "other"
    with pytest.raises(DomainError) as failure:
        service.import_context(other, context_for(), "global-context-collision")
    assert failure.value.code == "conflict"
    other_plant_registry = service.register_model(principal, model, SECOND_PLANT, "model-beta")
    assert other_plant_registry["object_id"] not in {second["object_id"], original_registry["object_id"]}
    with pytest.raises(DomainError):
        service.get_model(other, SECOND_PLANT)
    forbidden = context_for(plant_id=SECOND_PLANT, context_id="beta-context")
    with pytest.raises(DomainError):
        service.import_context(other, forbidden, "forbidden-import")


def test_operator_can_import_but_cannot_register_model(env, model):
    store, _, service, _ = env
    operator = store.create_user("operator", "c" * 32, "operator", [PLANT])
    assert service.import_context(operator, context_for(), "operator-import")["owner"] == "operator"
    with pytest.raises(DomainError):
        service.register_model(operator, model, PLANT, "operator-model")
    assert service.get_model(operator, PLANT) is None


def test_model_registration_receipts_and_registry_version(env, model):
    store, principal, service, _ = env
    first = service.register_model(principal, model, PLANT, "register")
    assert service.register_model(principal, model, PLANT, "register") == first
    replacement = service.register_model(principal, model, PLANT, "reregister")
    assert replacement["version"] == 2
    assert replacement["content_hash"] == first["content_hash"]
    assert service.register_model(principal, model, PLANT, "register") == first
    assert service.get_model(principal, PLANT) == replacement
    assert len(store.audit_events(principal)) == 2
    with pytest.raises(DomainError) as failure:
        service.register_model(principal, model, SECOND_PLANT, "register")
    assert failure.value.code == "idempotency_conflict"
    assert service.get_model(principal, SECOND_PLANT) is None


def test_long_owner_and_plant_use_stable_bounded_registry_identifier(env, model):
    store, _, service, clock = env
    plant = "p" * 80
    principal = store.create_user("u" * 80, "d" * 32, "reviewer", [plant])
    registered = service.register_model(principal, model, plant, "register")
    assert registered["object_id"] == "active-model-" + digest({"owner": principal.user_id, "plant_id": plant})
    assert len(registered["object_id"]) == 77
    reopened = RecommendationService(Store(store.root), clock=clock)
    assert reopened.get_model(principal, plant) == registered
    assert reopened.register_model(principal, model, plant, "register") == registered
    with pytest.raises(DomainError) as failure:
        service.register_model(principal, model, plant + "x", "invalid-plant")
    assert failure.value.status == 422


def test_snapshot_missing_model_is_explicit(env):
    _, principal, service, _ = env
    obj = import_one(env)
    snapshot = service.snapshot(principal, obj["object_id"])
    assert snapshot["context_hash"] == digest(obj["payload"])
    assert snapshot["context_version"] == 1
    assert snapshot["constraint_version"] == "synthetic-bop-v1"
    assert snapshot["experience_version"] == EXPERIENCE_VERSION
    for key in ("model", "model_hash", "model_content_hash", "model_version"):
        assert snapshot[key] is None


@pytest.mark.parametrize("corruption", ["missing_artifact", "artifact_bytes", "invalid_model_schema"])
def test_existing_corrupt_model_never_looks_missing(env, model, corruption):
    store, principal, service, _ = env
    obj = import_one(env)
    registered = service.register_model(principal, model, PLANT, "register")
    artifact = store.root / "artifacts" / (registered["content_hash"] + ".json")
    if corruption == "missing_artifact":
        artifact.unlink()
    elif corruption == "artifact_bytes":
        artifact.write_bytes(b"{}")
    else:
        update(env, "model", registered["object_id"], lambda p: p["model"].update(model_hash="0" * 64))
    for read in (lambda: service.get_model(principal, PLANT),
                 lambda: service.snapshot(principal, obj["object_id"])):
        with pytest.raises(DomainError) as failure:
            read()
        assert failure.value.code == "integrity"


@pytest.mark.parametrize("change", [
    {"context_id": "different-id"}, {"plant_id": SECOND_PLANT}, {"synthetic": False},
    {"unexpected": "payload"},
])
def test_context_semantic_tamper_with_valid_artifact_hash_is_rejected(env, change):
    _, principal, service, _ = env
    obj = import_one(env)
    update(env, "context", obj["object_id"], lambda p: p.update(change))
    with pytest.raises(DomainError) as failure:
        service.get_context(principal, obj["object_id"])
    assert failure.value.code == "integrity"


def test_experience_is_real_metadata_retrieval_with_eight_original_cards(env):
    _, principal, service, _ = env
    identifiers = set()
    for defect in ("short_shot", "flash", "warpage", "normal"):
        context = context_for(defect, machine_id=f"cell-{defect}")
        service.import_context(principal, context, f"import-{defect}")
        result = service.experience(principal, context.context_id)
        assert result["query"] == {"defect": defect, "material_id": context.material_id,
                                   "process_version": context.process_version}
        assert result["context_hash"] == digest(context.model_dump(mode="json"))
        assert len(result["items"]) == 2
        for item in result["items"]:
            identifiers.add(item["id"])
            assert item["version"] == EXPERIENCE_VERSION
            assert "synthetic" in item["provenance"] and "not production" in item["provenance"]
    assert len(identifiers) == 8


@pytest.mark.parametrize("changes", [
    {"material_id": "unseen-resin"}, {"process_version": "unseen-process"},
])
def test_unrelated_experience_is_empty(env, changes):
    _, principal, service, _ = env
    context = context_for(**changes)
    import_one(env, context)
    assert service.experience(principal, context.context_id)["items"] == []


def test_supplement_only_missing_values_keeps_timestamp_and_rebinds_device(env):
    store, principal, service, clock = env
    context = context_for("missing", observed_at_ms=NOW - MAX_CONTEXT_AGE_MS - 100)
    obj = import_one(env, context)
    clock.now += 10_000
    updated = supplement(env, obj["object_id"], obj["content_hash"],
                         {"material_viscosity": 1.05, "ambient_c": 23.0})
    assert updated["version"] == 2 and updated["payload"]["revision"] == 2
    assert updated["payload"]["observed_at_ms"] == context.observed_at_ms
    assert updated["payload"]["current"] == obj["payload"]["current"]
    assert "context_expired" in validate_context(ProcessContext.model_validate(updated["payload"]), clock())
    device = store.get_object("device", context.machine_id, principal)
    assert device["payload"]["context_hash"] == updated["content_hash"]
    assert device["payload"]["revision"] == 1
    assert device["payload"]["updated_at_ms"] == NOW
    assert device["version"] == 2
    event = next(e for e in store.audit_events(principal) if e["action"] == "supplement_context")
    assert event["detail"]["note"] == "Explicit synthetic measurement"
    assert service.get_context(principal, obj["object_id"]) == updated


@pytest.mark.parametrize("fields", [
    {}, {"observed_at_ms": NOW}, {"severity": 0.8}, {"ambient_c": None},
    {"ambient_c": True}, {"ambient_c": "23"}, {"ambient_c": float("nan")},
    {"ambient_c": float("inf")}, {"ambient_c": 4.9}, {"ambient_c": 45.1},
    {"material_viscosity": 0.29}, {"material_viscosity": 2.51}, {"ambient_c": 10**400},
])
def test_supplement_rejects_unsupported_or_invalid_values_atomically(env, fields):
    store, principal, _, _ = env
    obj = import_one(env, context_for("missing"))
    with pytest.raises(DomainError):
        supplement(env, obj["object_id"], obj["content_hash"], fields)
    assert store.get_object("context", obj["object_id"], principal) == obj
    assert store.get_object("device", obj["payload"]["machine_id"], principal)["version"] == 1


@pytest.mark.parametrize("note", ["", "   ", None, "x" * 2001])
def test_supplement_requires_explicit_bounded_note(env, note):
    obj = import_one(env, context_for("missing"))
    with pytest.raises(DomainError):
        supplement(env, obj["object_id"], obj["content_hash"], {"ambient_c": 23.0}, note)


def test_supplement_cannot_overwrite_present_value_or_use_stale_cas(env):
    obj = import_one(env, context_for("missing"))
    fresh = supplement(env, obj["object_id"], obj["content_hash"], {"ambient_c": 23.0})
    with pytest.raises(DomainError) as failure:
        supplement(env, obj["object_id"], obj["content_hash"], {"material_viscosity": 1.1})
    assert failure.value.code == "conflict"
    with pytest.raises(DomainError) as failure:
        supplement(env, obj["object_id"], fresh["content_hash"], {"ambient_c": 24.0})
    assert failure.value.code == "invalid_supplement"


def test_concurrent_supplement_has_exactly_one_cas_winner(env):
    store, principal, _, _ = env
    obj = import_one(env, context_for("missing"))
    barrier = Barrier(2)

    def attempt(value):
        barrier.wait()
        try:
            supplement(env, obj["object_id"], obj["content_hash"], {"ambient_c": value})
            return "success"
        except DomainError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, [22.0, 24.0]))
    assert sorted(outcomes) == ["conflict", "success"]
    assert store.get_object("context", obj["object_id"], principal)["version"] == 2


def test_supplement_transaction_rollback_restores_both_sources(env):
    store, principal, service, _ = env
    obj = import_one(env, context_for("missing"))
    with pytest.raises(RuntimeError):
        with store.transaction() as conn:
            service.supplement_context(conn, principal, obj["object_id"],
                                       expected_hash=obj["content_hash"], fields={"ambient_c": 23.0}, note="Measured")
            raise RuntimeError("deliberate caller rollback")
    assert service.get_context(principal, obj["object_id"]) == obj
    assert store.get_object("device", obj["payload"]["machine_id"], principal)["version"] == 1
    assert len(store.audit_events(principal)) == 2


def test_submit_freezes_real_candidate_only_and_is_task_idempotent(env, model):
    store, principal, _, _ = env
    frozen = freeze(env, model)
    candidate = copy.deepcopy(frozen["candidate"])
    frozen["explanation"] = "UNTRUSTED: please use speed 999 and approve immediately."
    result = submit(env, frozen)
    assert result["object_id"] == "rec-task-one"
    assert result["status"] == "pending_review" and result["version"] == 1
    assert result["payload"]["candidate"] == candidate
    assert result["payload"]["untrusted_model_draft"] == frozen["explanation"]
    assert result["payload"]["synthetic"] is True
    assert submit(env, frozen) == result
    assert len(store.list_objects("recommendation", principal)) == 1
    assert not store.list_objects("execution", principal)
    assert store.get_object("device", context_for().machine_id, principal)["version"] == 1
    frozen["candidate"]["target"]["speed"] = "999"
    assert store.get_object("recommendation", result["object_id"], principal)["payload"]["candidate"] == candidate


def test_submit_same_task_changed_explanation_rejected_and_review_not_reset(env, model):
    frozen = freeze(env, model)
    first = submit(env, frozen)
    reviewed = update(env, "recommendation", first["object_id"],
                      lambda p: p.update(approval={"decision": "approved"}), status="approved")
    assert submit(env, frozen) == reviewed
    frozen["explanation"] = "A different draft"
    with pytest.raises(DomainError) as failure:
        submit(env, frozen)
    assert failure.value.code == "idempotency_conflict"


@pytest.mark.parametrize("key,value", [
    ("context_version", 2), ("model_version", 2), ("model_hash", "0" * 64),
    ("model_content_hash", "0" * 64), ("context_hash", "0" * 64),
    ("constraint_version", "other-bop"), ("experience_version", "other-experience"),
    ("context_version", True), ("model_version", 1.0),
])
def test_submit_revalidates_each_exact_snapshot_identity(env, model, key, value):
    frozen = freeze(env, model)
    frozen["expected_snapshot"][key] = value
    with pytest.raises(DomainError) as failure:
        submit(env, frozen)
    assert failure.value.code == "source_changed"


def test_registering_same_model_with_new_action_invalidates_old_task(env, model):
    _, principal, service, _ = env
    frozen = freeze(env, model)
    service.register_model(principal, model, PLANT, "replace-registry")
    with pytest.raises(DomainError) as failure:
        submit(env, frozen)
    assert failure.value.code == "source_changed"


def test_model_replacement_with_new_hash_invalidates_old_task(env, model):
    _, principal, service, _ = env
    frozen = freeze(env, model)
    replacement = bc.train_model(generate_demonstrations(groups=20, per_group=12), seed=123)["model"]
    assert replacement["model_hash"] != model["model_hash"]
    service.register_model(principal, replacement, PLANT, "new-model")
    with pytest.raises(DomainError) as failure:
        submit(env, frozen)
    assert failure.value.code == "source_changed"


def test_identical_context_content_new_object_version_invalidates_task(env, model):
    frozen = freeze(env, model)
    update(env, "context", frozen["context_id"], lambda p: None)
    with pytest.raises(DomainError) as failure:
        submit(env, frozen)
    assert failure.value.code == "source_changed"


@pytest.mark.parametrize("elapsed", [MAX_CONTEXT_AGE_MS + 1, -2000])
def test_submit_rechecks_expired_and_future_observation_without_hash_change(env, model, elapsed):
    *_, clock = env
    frozen = freeze(env, model)
    clock.now += elapsed
    with pytest.raises(DomainError) as failure:
        submit(env, frozen)
    assert failure.value.code == "candidate_ineligible"


@pytest.mark.parametrize("field,value", [
    ("candidate_hash", "0" * 64), ("target", {"temperature": "215", "speed": "100", "pressure": "60", "cooling": "19"}),
    ("task_id", "other-task"), ("context_id", "other-context"), ("model_hash", "0" * 64),
    ("eligible", 1), ("extra", "not-frozen"),
])
def test_candidate_independent_tamper_and_rehash_is_rejected(env, model, field, value):
    frozen = freeze(env, model)
    frozen["candidate"][field] = value
    if field != "candidate_hash":
        frozen["candidate"]["candidate_hash"] = digest({k: v for k, v in frozen["candidate"].items()
                                                        if k != "candidate_hash"})
    with pytest.raises(DomainError) as failure:
        submit(env, frozen)
    assert failure.value.code == "candidate_changed"


def test_raw_delta_over_limit_is_not_laundered_by_frozen_rounded_target(env, model):
    frozen = freeze(env, model)
    frozen["candidate"]["raw_delta"]["pressure"] = "-4.1"
    frozen["candidate"]["candidate_hash"] = digest({k: v for k, v in frozen["candidate"].items()
                                                    if k != "candidate_hash"})
    with pytest.raises(DomainError) as failure:
        submit(env, frozen)
    assert failure.value.code == "candidate_ineligible"


def test_eligible_candidate_not_produced_by_bc_rejected_even_if_task_rehashed(env, model):
    frozen = freeze(env, model)
    context = context_for()
    fabricated = check_candidate(context, {"temperature": "0", "speed": "-5", "pressure": "0", "cooling": "0"}, NOW)
    assert fabricated["eligible"]
    fabricated.update(task_id=frozen["task_id"], context_id=frozen["context_id"], model_hash=model["model_hash"])
    fabricated["candidate_hash"] = digest(fabricated)
    frozen["candidate"] = fabricated
    update(env, "task", frozen["task_id"], lambda p: p.update(candidate=fabricated))
    with pytest.raises(DomainError) as failure:
        submit(env, frozen)
    assert failure.value.code == "prediction_changed"


@pytest.mark.parametrize("change", ["hash", "payload", "task", "context", "duplicate_id", "duplicate_tool", "missing", "extra"])
def test_evidence_independent_tamper_rejected(env, model, change):
    frozen = freeze(env, model)
    evidence = frozen["evidence"]
    item = evidence[0]
    if change == "hash":
        item["content_hash"] = "0" * 64
    elif change == "payload":
        item["payload"]["context"]["severity"] = 0.91
    elif change == "task":
        item["task_id"] = "different-task"
    elif change == "context":
        item["context_hash"] = "0" * 64
    elif change == "duplicate_id":
        item["evidence_id"] = evidence[1]["evidence_id"]
    elif change == "duplicate_tool":
        evidence[0] = copy.deepcopy(evidence[1])
        evidence[0]["evidence_id"] = "unique-id"
        item = evidence[0]
    elif change == "missing":
        evidence.pop()
    else:
        evidence.append(copy.deepcopy(evidence[0]))
    if change != "hash":
        item["content_hash"] = digest({k: v for k, v in item.items() if k != "content_hash"})
    with pytest.raises(DomainError) as failure:
        submit(env, frozen)
    assert failure.value.code == "invalid_evidence"


def test_evidence_payload_recomputed_independently_of_rehashed_stored_task(env, model):
    frozen = freeze(env, model)
    item = frozen["evidence"][1]
    item["payload"]["items"][0]["text"] = "Fabricated production approval"
    item["content_hash"] = digest({k: v for k, v in item.items() if k != "content_hash"})
    update(env, "task", frozen["task_id"], lambda p: p.update(evidence=frozen["evidence"]))
    with pytest.raises(DomainError) as failure:
        submit(env, frozen)
    assert failure.value.code == "invalid_evidence"


def test_self_excluding_hashes_and_backend_issuance_are_required(env, model):
    frozen = freeze(env, model)
    assert frozen["candidate"]["candidate_hash"] == digest({k: v for k, v in frozen["candidate"].items()
                                                           if k != "candidate_hash"})
    for item in frozen["evidence"]:
        assert item["content_hash"] == digest({k: v for k, v in item.items() if k != "content_hash"})
    assert submit(env, frozen)["status"] == "pending_review"
    # A hash is not authority: even a correct recomputed result must be issued by
    # the backend and referenced in the persisted task rather than minted by text.
    frozen["evidence"][0]["evidence_id"] = "model-minted-id"
    item = frozen["evidence"][0]
    item["content_hash"] = digest({k: v for k, v in item.items() if k != "content_hash"})
    with pytest.raises(DomainError) as failure:
        submit(env, frozen)
    assert failure.value.code == "invalid_evidence"


@pytest.mark.parametrize("status", ["created", "cancelled", "blocked", "awaiting_input", "submitted"])
def test_stopped_task_cannot_create_recommendation(env, model, status):
    frozen = freeze(env, model)
    update(env, "task", frozen["task_id"], lambda p: p.update(status=status), status=status)
    with pytest.raises(DomainError) as failure:
        submit(env, frozen)
    assert failure.value.code == "task_stopped"


def test_cross_owner_submit_and_cross_plant_task_binding_rejected(env, model):
    store, principal, service, _ = env
    frozen = freeze(env, model)
    other = store.create_user("other", "b" * 32, "reviewer", [PLANT])
    with pytest.raises(DomainError) as failure:
        with store.transaction() as conn:
            service.submit_recommendation(conn, other, **frozen)
    assert failure.value.status == 404
    with store.transaction() as conn:
        # Test semantically inconsistent metadata independently of artifact hash.
        conn.execute("UPDATE objects SET plant_id=? WHERE kind='task' AND object_id=? AND owner=?",
                     (SECOND_PLANT, frozen["task_id"], principal.user_id))
    with pytest.raises(DomainError) as failure:
        submit(env, frozen)
    assert failure.value.code == "source_changed"


@pytest.mark.parametrize("operation", ["get", "snapshot", "experience", "import-replay", "model-replay", "submit"])
def test_revoked_principal_cannot_read_write_or_replay(env, model, operation):
    store, principal, service, _ = env
    frozen = freeze(env, model)
    if operation == "submit":
        submit(env, frozen)
    store.set_user(principal.user_id, active=False)
    calls = {
        "get": lambda: service.get_context(principal, frozen["context_id"]),
        "snapshot": lambda: service.snapshot(principal, frozen["context_id"]),
        "experience": lambda: service.experience(principal, frozen["context_id"]),
        "import-replay": lambda: service.import_context(principal, context_for(), "import-1"),
        "model-replay": lambda: service.register_model(principal, model, PLANT, "register-1"),
        "submit": lambda: submit(env, frozen),
    }
    with pytest.raises(DomainError) as failure:
        calls[operation]()
    assert failure.value.status in (401, 403)


def test_supplement_after_acl_change_rejected(env):
    store, principal, _, _ = env
    obj = import_one(env, context_for("missing"))
    store.set_user(principal.user_id, plants=[SECOND_PLANT])
    with pytest.raises(DomainError):
        supplement(env, obj["object_id"], obj["content_hash"], {"ambient_c": 23.0})


def test_submit_rollback_creates_neither_recommendation_nor_audit(env, model):
    store, principal, service, _ = env
    frozen = freeze(env, model)
    before = canonical(store.audit_events(principal))
    with pytest.raises(RuntimeError):
        with store.transaction() as conn:
            service.submit_recommendation(conn, principal, **frozen)
            raise RuntimeError("deliberate caller rollback")
    assert store.list_objects("recommendation", principal) == []
    assert canonical(store.audit_events(principal)) == before


def test_overlong_task_id_rejected(env, model):
    frozen = freeze(env, model)
    frozen["task_id"] = "t" * 61
    with pytest.raises(DomainError) as failure:
        submit(env, frozen)
    assert failure.value.status == 422


def test_import_replay_after_device_advance_does_not_restore_old_state(env):
    store, principal, service, _ = env
    original = import_one(env)
    machine = original["payload"]["machine_id"]
    advanced = update(env, "device", machine,
                      lambda p: p.update(current={**p["current"], "speed": "70"}, revision=2))
    assert service.import_context(principal, context_for(), "import-1") == original
    assert store.get_object("device", machine, principal) == advanced


def test_concurrent_duplicate_submit_returns_one_immutable_recommendation(env, model):
    store, principal, _, _ = env
    frozen = freeze(env, model)
    barrier = Barrier(2)

    def attempt(_):
        barrier.wait()
        return submit(env, copy.deepcopy(frozen))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert results[0] == results[1]
    assert len(store.list_objects("recommendation", principal)) == 1
    assert sum(e["action"] == "submit_recommendation" for e in store.audit_events(principal)) == 1


def test_corrupt_model_blocks_submission_even_when_task_hashes_are_intact(env, model):
    _, principal, service, _ = env
    frozen = freeze(env, model)
    registered = service.get_model(principal, PLANT)
    update(env, "model", registered["object_id"],
           lambda p: p["model"]["training"].update(epochs=0))
    with pytest.raises(DomainError) as failure:
        submit(env, frozen)
    assert failure.value.code == "integrity"


def test_cross_operation_action_id_is_not_reusable(env, model):
    _, principal, service, _ = env
    original = import_one(env)
    with pytest.raises(DomainError) as failure:
        service.register_model(principal, model, PLANT, "import-1")
    assert failure.value.code == "idempotency_conflict"
    assert service.get_model(principal, PLANT) is None
    assert service.get_context(principal, original["object_id"]) == original


def test_current_acl_and_role_are_rechecked_before_model_receipt_replay(env, model):
    store, principal, service, _ = env
    service.register_model(principal, model, PLANT, "register")
    store.set_user(principal.user_id, role="operator")
    refreshed = store.authenticate("a" * 32)
    with pytest.raises(DomainError):
        service.register_model(refreshed, model, PLANT, "register")


def test_valid_import_idempotency_does_not_make_new_context_alias(env):
    _, principal, service, _ = env
    import_one(env)
    new_context = context_for(context_id="different-context", machine_id="different-cell")
    with pytest.raises(DomainError) as failure:
        service.import_context(principal, new_context, "import-1")
    assert failure.value.code == "idempotency_conflict"
    with pytest.raises(DomainError) as failure:
        service.get_context(principal, "different-context")
    assert failure.value.status == 404


def test_supplement_does_not_make_model_abstention_disappear_for_stale_data(env, model):
    _, principal, service, clock = env
    context = context_for("missing", observed_at_ms=NOW - MAX_CONTEXT_AGE_MS - 1)
    original = import_one(env, context)
    service.register_model(principal, model, PLANT, "register")
    updated = supplement(env, original["object_id"], original["content_hash"],
                         {"material_viscosity": 1.05, "ambient_c": 23.0})
    prediction = bc.predict(model, ProcessContext.model_validate(updated["payload"]), clock())
    assert prediction["status"] == "abstain"
    assert "context_expired" in prediction["issues"]


def test_semantic_context_defaults_cannot_launder_missing_stored_fields(env):
    _, principal, service, _ = env
    obj = import_one(env, context_for("missing"))
    update(env, "context", obj["object_id"], lambda p: p.pop("ambient_c"))
    with pytest.raises(DomainError) as failure:
        service.get_context(principal, obj["object_id"])
    assert failure.value.code == "integrity"


@pytest.mark.parametrize("change", [
    {"revision": True}, {"revision": 0}, {"updated_at_ms": None},
    {"updated_at_ms": -1}, {"context_hash": "0" * 64},
])
def test_supplement_revalidates_device_and_rolls_back_context(env, change):
    _, principal, service, _ = env
    obj = import_one(env, context_for("missing"))
    update(env, "device", obj["payload"]["machine_id"], lambda p: p.update(change))
    with pytest.raises(DomainError) as failure:
        supplement(env, obj["object_id"], obj["content_hash"], {"ambient_c": 23.0})
    assert failure.value.code == "device_conflict"
    assert service.get_context(principal, obj["object_id"]) == obj
