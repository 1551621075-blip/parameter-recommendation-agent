"""Actual SQLite/action/domain/model/simulator integration, entirely temporary.

The module trains one small real BC artifact. Fixtures create authentic, issued
source-bound tool results through RecommendationService's submission boundary;
no safety decision, constraint gate, authentication, or simulator is mocked.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier

import pytest

from parameter_agent import bc
from parameter_agent.actions import ActionService
from parameter_agent.domain import ProcessContext, check_candidate, digest, validate_context
from parameter_agent.errors import DomainError
from parameter_agent.fixtures import generate_demonstrations, simulate_quality
from parameter_agent.service import SOURCE_KEYS, RecommendationService
from parameter_agent.storage import Store


@dataclass
class Clock:
    value: int

    def __call__(self):
        return self.value


@pytest.fixture(scope="module")
def trained():
    records = generate_demonstrations(groups=10, per_group=8)
    result = bc.train_model(records)
    model = result["model"]
    now = max(row["context"]["observed_at_ms"] for row in records) + 1000
    for row in records:
        context = ProcessContext.model_validate(row["context"])
        prediction = bc.predict(model, context, now)
        if prediction["status"] == "ok":
            assert check_candidate(context, prediction["raw_delta"], now)["eligible"] is True
            return model, context, now
    pytest.fail("The genuinely trained small BC fixture has no eligible predictions")


class Scenario:
    def __init__(self, root, model, context, now):
        self.root = root
        self.model = model
        self.context = context
        self.clock = Clock(now)
        self.store = Store(root)
        self.token = "".join(("test-local-reviewer-", "a" * 40))
        self.principal = self.store.create_user(
            "synthetic-reviewer", self.token, "reviewer", [context.plant_id]
        )
        self.service = RecommendationService(self.store, clock=self.clock)
        self.actions = ActionService(self.service, clock=self.clock)
        self.service.import_context(self.principal, context, "import-context")
        self.service.register_model(self.principal, model, context.plant_id, "register-model")
        self.recommendation = self.submit("synthetic-task-action")

    def submit(self, task_id):
        with self.store.transaction() as conn:
            snapshot = self.service.snapshot(self.principal, self.context.context_id, conn=conn)
            context = ProcessContext.model_validate(snapshot["context"])
            source = {key: snapshot[key] for key in SOURCE_KEYS}
            prediction = bc.predict(snapshot["model"], context, self.clock())
            assert prediction["status"] == "ok"
            candidate = check_candidate(context, prediction["raw_delta"], self.clock())
            candidate.update(
                task_id=task_id, context_id=context.context_id, model_hash=snapshot["model_hash"]
            )
            candidate["candidate_hash"] = digest(candidate)
            results = {
                "get_process_context": {
                    "context": snapshot["context"],
                    "issues": validate_context(context, self.clock()),
                },
                "retrieve_experience": self.service._experience(snapshot["context"]),
                "predict_bc": prediction,
                "check_constraints": {"candidate": candidate},
            }
            evidence = []
            for index, (tool, payload) in enumerate(results.items()):
                item = {
                    "evidence_id": f"evidence-{task_id}-{index}",
                    "task_id": task_id,
                    "tool": tool,
                    "context_hash": snapshot["context_hash"],
                    "payload": payload,
                }
                item["content_hash"] = digest(item)
                evidence.append(item)
            self.store.create_object(
                conn,
                kind="task",
                object_id=task_id,
                principal=self.principal,
                plant_id=context.plant_id,
                status="running",
                payload={
                    "status": "running",
                    "context_id": context.context_id,
                    "source_snapshot": source,
                    "prediction": prediction,
                    "candidate": candidate,
                    "evidence": evidence,
                },
            )
            return self.service.submit_recommendation(
                conn,
                self.principal,
                task_id=task_id,
                context_id=context.context_id,
                expected_snapshot=source,
                candidate=candidate,
                evidence=evidence,
                explanation="Untrusted explanation for an original synthetic trained candidate.",
            )

    def get(self, kind, identifier=None):
        if identifier is None:
            identifier = (
                self.context.machine_id if kind == "device" else self.recommendation["object_id"]
            )
        return self.store.get_object(kind, identifier, self.principal)

    def mutate(self, kind, object_id, modify, *, status=None):
        """Authorized low-level mutation used to test drift, not bypass a gate."""
        with self.store.transaction() as conn:
            obj = self.store.get_object(kind, object_id, self.principal, conn=conn)
            payload = copy.deepcopy(obj["payload"])
            modify(payload)
            return self.store.update_object(
                conn,
                kind=kind,
                object_id=object_id,
                principal=self.principal,
                expected_version=obj["version"],
                status=status or obj["status"],
                payload=payload,
            )

    def approve(self, *, action_id="review-action", recommendation=None):
        recommendation = recommendation or self.recommendation
        return self.actions.review(
            self.principal,
            recommendation["object_id"],
            action_id=action_id,
            recommendation_hash=recommendation["content_hash"],
            decision="approved",
            note="Synthetic review only.",
        )

    def request(self, *, recommendation=None, action_id="request-action"):
        recommendation = recommendation or self.approve()
        return self.actions.request_execution(
            self.principal,
            recommendation["object_id"],
            action_id=action_id,
            recommendation_hash=recommendation["content_hash"],
            confirmed=True,
        )

    def apply(self, *, fault=None):
        execution = self.request()
        return self.actions.apply_synthetic(
            self.principal, execution["object_id"], action_id="apply-action", fault=fault
        )

    def readback(self):
        execution = self.apply()
        return self.actions.readback(
            self.principal, execution["object_id"], action_id="readback-action"
        )

    def observe(self, **kwargs):
        return self.actions.observe(
            self.principal, self.recommendation["object_id"], action_id="observe-action", **kwargs
        )

    def restart(self):
        self.store = Store(self.root)
        self.principal = self.store.authenticate(self.token)
        self.service = RecommendationService(self.store, clock=self.clock)
        self.actions = ActionService(self.service, clock=self.clock)


@pytest.fixture
def scenario(tmp_path, trained):
    model, context, now = trained
    return Scenario(tmp_path.resolve() / "local-synthetic", model, context, now)


def assert_error(code, function, *args, **kwargs):
    with pytest.raises(DomainError) as error:
        function(*args, **kwargs)
    assert error.value.code == code


def test_review_request_apply_readback_are_distinct_facts(scenario):
    s = scenario
    original = s.get("device")
    approved = s.approve()
    assert approved["status"] == "approved"
    assert (
        approved["payload"]["approval"]["recommendation_hash"] == s.recommendation["content_hash"]
    )
    assert s.get("device") == original
    assert s.store.list_objects("execution", s.principal) == []
    execution = s.request(recommendation=approved)
    assert execution["status"] == "pending"
    assert s.get("recommendation")["status"] == "execution_pending"
    assert s.get("device") == original
    assert s.store.list_objects("device_receipt", s.principal) == []
    assert_error("invalid_state", s.observe)
    assert_error(
        "invalid_state",
        s.actions.readback,
        s.principal,
        execution["object_id"],
        action_id="too-soon",
    )
    applied = s.actions.apply_synthetic(
        s.principal, execution["object_id"], action_id="apply-action"
    )
    assert applied["status"] == "applied"
    assert "readback" not in applied["payload"]
    assert s.get("recommendation")["status"] == "execution_pending"
    device = s.get("device")
    assert device["version"] == original["version"] + 1
    assert device["payload"]["revision"] == original["payload"]["revision"] + 1
    assert device["payload"]["current"] == execution["payload"]["target"]
    assert device["payload"]["current"] != original["payload"]["current"]
    receipt = s.get("device_receipt", execution["object_id"])
    assert receipt["payload"]["before"] == original["payload"]["current"]
    assert receipt["payload"]["target"] == device["payload"]["current"]
    assert receipt["payload"]["request_hash"] == execution["payload"]["request_hash"]
    assert_error("invalid_state", s.observe)
    done = s.actions.readback(s.principal, execution["object_id"], action_id="readback-action")
    assert done["status"] == "executed"
    assert s.get("recommendation")["status"] == "executed"
    assert done["payload"]["readback"]["device_hash"] == device["content_hash"]
    assert s.get("device") == device
    actions = {event["action"] for event in s.store.audit_events(s.principal)}
    assert {
        "review",
        "request_execution",
        "synthetic_apply",
        "apply_synthetic",
        "readback_verified",
    } <= actions


@pytest.mark.parametrize("confirmed", [False, None, 1, 1.0, "true", [], {}])
def test_execution_requires_literal_true(scenario, confirmed):
    s = scenario
    approved = s.approve()
    assert_error(
        "confirmation_required",
        s.actions.request_execution,
        s.principal,
        approved["object_id"],
        action_id="request-action",
        recommendation_hash=approved["content_hash"],
        confirmed=confirmed,
    )
    assert s.get("recommendation")["status"] == "approved"
    assert s.get("device")["version"] == 1
    assert s.store.list_objects("execution", s.principal) == []


@pytest.mark.parametrize(
    "decision,note",
    [
        (True, "note"),
        (1, "note"),
        ("approve", "note"),
        ("approved", " "),
        ("approved", 1),
        ("approved", "n" * 4001),
    ],
)
def test_review_input_is_strict(scenario, decision, note):
    s = scenario
    with pytest.raises(DomainError):
        s.actions.review(
            s.principal,
            s.recommendation["object_id"],
            action_id="review",
            recommendation_hash=s.recommendation["content_hash"],
            decision=decision,
            note=note,
        )
    assert s.get("recommendation")["status"] == "pending_review"


@pytest.mark.parametrize("bad_hash", [None, True, 123, "", "A" * 64, "x" * 64])
def test_review_hash_format_is_strict(scenario, bad_hash):
    s = scenario
    assert_error(
        "invalid_hash",
        s.actions.review,
        s.principal,
        s.recommendation["object_id"],
        action_id="review",
        recommendation_hash=bad_hash,
        decision="approved",
        note="note",
    )


def test_review_exact_hash_and_rejection_are_terminal(scenario):
    s = scenario
    assert_error(
        "action_integrity",
        s.actions.review,
        s.principal,
        s.recommendation["object_id"],
        action_id="review",
        recommendation_hash="0" * 64,
        decision="approved",
        note="note",
    )
    rejected = s.actions.review(
        s.principal,
        s.recommendation["object_id"],
        action_id="reject",
        recommendation_hash=s.recommendation["content_hash"],
        decision="rejected",
        note="No synthetic actuation.",
    )
    assert rejected["status"] == "rejected"
    assert_error("invalid_state", s.request, recommendation=rejected)
    assert_error("invalid_state", s.approve, action_id="change-mind", recommendation=rejected)
    assert s.get("device")["version"] == 1


def test_execution_uses_hash_after_approval_not_original(scenario):
    s = scenario
    approved = s.approve()
    assert approved["content_hash"] != s.recommendation["content_hash"]
    assert_error("action_integrity", s.request, recommendation=s.recommendation)
    assert s.request(recommendation=approved)["status"] == "pending"


@pytest.mark.parametrize(
    "change",
    [
        "context_version",
        "context_content",
        "model_version",
        "candidate",
        "device_version",
        "device_revision",
        "device_current",
        "device_context",
        "device_process",
        "device_synthetic",
        "device_time",
    ],
)
def test_execution_request_rechecks_sources_constraints_and_device(scenario, change):
    s = scenario
    approved = s.approve()
    if change.startswith("context"):
        s.mutate(
            "context",
            s.context.context_id,
            lambda p: p.update(severity=0.73) if change == "context_content" else None,
        )
    elif change == "model_version":
        s.service.register_model(s.principal, s.model, s.context.plant_id, "new-model-version")
    elif change == "candidate":

        def tamper(p):
            p["candidate"]["target"]["temperature"] = "240"
            p["candidate"]["candidate_hash"] = digest(
                {k: v for k, v in p["candidate"].items() if k != "candidate_hash"}
            )

        approved = s.mutate("recommendation", approved["object_id"], tamper)
    else:

        def drift(p):
            if change == "device_revision":
                p["revision"] += 1
            elif change == "device_current":
                p["current"]["pressure"] = "31"
            elif change == "device_context":
                p["context_id"] = "synthetic-different-context"
            elif change == "device_process":
                p["process_version"] = "synthetic-process-v2"
            elif change == "device_synthetic":
                p["synthetic"] = 1
            elif change == "device_time":
                p["updated_at_ms"] = s.clock() + 1

        s.mutate("device", s.context.machine_id, drift)
    if change in {"device_version", "device_revision"}:
        # A request binds the CURRENT actual revision; no earlier device version
        # is invented from the context revision. Application must keep that bind.
        execution = s.request(recommendation=approved)
        assert execution["payload"]["expected_device_version"] == s.get("device")["version"]
        return
    with pytest.raises(DomainError):
        s.request(recommendation=approved)
    assert s.store.list_objects("execution", s.principal) == []


@pytest.mark.parametrize("phase", ["request", "apply"])
def test_context_must_still_be_fresh_at_request_and_application(scenario, phase):
    s = scenario
    obj = s.approve() if phase == "request" else s.request()
    s.clock.value = s.context.observed_at_ms + 86_400_001
    if phase == "request":
        assert_error("unsafe_candidate", s.request, recommendation=obj)
    else:
        assert_error(
            "unsafe_candidate",
            s.actions.apply_synthetic,
            s.principal,
            obj["object_id"],
            action_id="apply",
        )
    assert s.get("device")["version"] == 1
    assert s.store.list_objects("device_receipt", s.principal) == []


@pytest.mark.parametrize("change", ["version", "revision", "current", "context_hash", "model"])
def test_application_rechecks_version_and_source_after_request(scenario, change):
    s = scenario
    execution = s.request()
    if change == "model":
        s.service.register_model(s.principal, s.model, s.context.plant_id, "replace-model")
    else:

        def drift(p):
            if change == "revision":
                p["revision"] += 1
            elif change == "current":
                p["current"]["cooling"] = "29"
            elif change == "context_hash":
                p["context_hash"] = "0" * 64

        s.mutate("device", s.context.machine_id, drift)
    before = s.get("device")
    with pytest.raises(DomainError):
        s.actions.apply_synthetic(s.principal, execution["object_id"], action_id="apply")
    assert s.get("device") == before
    assert s.get("execution", execution["object_id"])["status"] == "pending"
    assert s.store.list_objects("device_receipt", s.principal) == []


def test_unknown_persists_committed_device_receipt_across_restart_no_blind_resend(scenario):
    s = scenario
    unknown = s.apply(fault="after_apply")
    identifier = unknown["object_id"]
    assert unknown["status"] == "unknown"
    assert unknown["payload"]["adapter_acknowledged"] is False
    device = s.get("device")
    receipt = s.get("device_receipt", identifier)
    assert device["payload"]["current"] == unknown["payload"]["target"]
    s.restart()
    assert s.get("execution", identifier) == unknown
    assert s.get("recommendation")["status"] == "unknown"
    assert s.get("device") == device
    assert s.get("device_receipt", identifier) == receipt
    replay = s.actions.apply_synthetic(
        s.principal, identifier, action_id="apply-action", fault="after_apply"
    )
    assert replay == unknown
    assert_error(
        "idempotency_conflict",
        s.actions.apply_synthetic,
        s.principal,
        identifier,
        action_id="apply-action",
    )
    assert_error(
        "invalid_state",
        s.actions.apply_synthetic,
        s.principal,
        identifier,
        action_id="blind-resend",
    )
    assert_error("invalid_state", s.observe)
    recovered = s.actions.readback(s.principal, identifier, action_id="recover")
    assert recovered["status"] == "executed"
    assert recovered["payload"]["readback"]["device_receipt_hash"] == receipt["content_hash"]
    assert s.get("device") == device
    assert len(s.store.list_objects("device_receipt", s.principal)) == 1


@pytest.mark.parametrize("start", ["applied", "unknown"])
@pytest.mark.parametrize(
    "damage",
    [
        "missing_receipt",
        "receipt_target",
        "receipt_before",
        "receipt_request",
        "receipt_revision",
        "receipt_source",
        "device_current",
        "device_revision",
        "device_context",
        "device_version",
    ],
)
def test_readback_requires_real_matching_receipt_and_device(scenario, start, damage):
    s = scenario
    execution = s.apply(fault="after_apply" if start == "unknown" else None)
    identifier = execution["object_id"]
    if damage == "missing_receipt":
        with s.store.transaction() as conn:
            conn.execute(
                "DELETE FROM objects WHERE kind=? AND object_id=? AND owner=?",
                ("device_receipt", identifier, s.principal.user_id),
            )
    elif damage.startswith("receipt"):

        def corrupt(p):
            if damage == "receipt_target":
                p["target"]["pressure"] = "31"
            elif damage == "receipt_before":
                p["before"]["pressure"] = "31"
            elif damage == "receipt_request":
                p["request_id"] = "different-request"
            elif damage == "receipt_revision":
                p["device_revision"] += 1
            elif damage == "receipt_source":
                p["source_hash"] = "0" * 64

        s.mutate("device_receipt", identifier, corrupt)
    else:

        def corrupt(p):
            if damage == "device_current":
                p["current"]["pressure"] = "31"
            elif damage == "device_revision":
                p["revision"] += 1
            elif damage == "device_context":
                p["context_id"] = "different-active-context"

        s.mutate("device", s.context.machine_id, corrupt)
    before = s.get("device")
    result = s.actions.readback(s.principal, identifier, action_id="readback-invalid")
    assert result["status"] == "unknown"
    assert result["payload"]["readback_attempt"]["verified"] is False
    assert "readback" not in result["payload"]
    assert s.get("recommendation")["status"] == "unknown"
    assert s.get("device") == before
    s.restart()
    assert s.get("execution", identifier)["status"] == "unknown"
    assert_error(
        "invalid_state",
        s.actions.apply_synthetic,
        s.principal,
        identifier,
        action_id="blind-resend",
    )


def test_readback_cannot_accept_a_human_success_claim(scenario):
    s = scenario
    execution = s.apply(fault="after_apply")
    with pytest.raises(TypeError):
        s.actions.readback(
            s.principal,
            execution["object_id"],
            action_id="made-up",
            success=True,
            current=execution["payload"]["target"],
        )
    assert s.get("execution", execution["object_id"])["status"] == "unknown"


@pytest.mark.parametrize("age", [-1, 0, 59_000, 59_999])
def test_observation_uses_actual_clock_age_after_readback(scenario, age):
    s = scenario
    s.readback()
    s.clock.value += age
    assert_error("observation_immature", s.observe)
    assert s.get("recommendation")["status"] == "executed"
    assert "observation" not in s.get("recommendation")["payload"]


def test_wait_before_readback_does_not_count_as_observation_maturity(scenario):
    s = scenario
    execution = s.apply()
    s.clock.value += 120_000
    s.actions.readback(s.principal, execution["object_id"], action_id="late-readback")
    assert_error("observation_immature", s.observe)
    s.clock.value += 60_000
    assert s.observe()["status"] == "observed"


@pytest.mark.parametrize("sample_count", [True, False, 60.0, "120", None, 0, 59, 10_001])
def test_observation_sample_size_is_strict(scenario, sample_count):
    s = scenario
    s.readback()
    s.clock.value += 60_000
    assert_error("invalid_sample_count", s.observe, sample_count=sample_count)
    assert s.get("recommendation")["status"] == "executed"


@pytest.mark.parametrize("seed", [True, 1.0, "1", -1, 2**63, None])
def test_observation_seed_is_strict(scenario, seed):
    s = scenario
    s.readback()
    s.clock.value += 60_000
    assert_error("invalid_seed", s.observe, seed=seed)


@pytest.mark.parametrize("sample_count", [60, 120, 10_000])
def test_observation_calculates_fresh_simulation_counts_intervals_and_honest_conclusion(
    scenario, sample_count
):
    s = scenario
    execution = s.readback()
    s.clock.value += 60_000
    observed = s.observe(sample_count=sample_count)
    observation = observed["payload"]["observation"]
    assert observed["status"] == "observed"
    assert observation["age_after_readback_ms"] == 60_000
    assert observation["causal_effect_proven"] is False
    assert observation["baseline_is_observed_production"] is False
    assert observation["paired_simulated_comparison"] is True
    assert observation["independent_samples"] is True
    assert observation["before_seed"] != observation["after_seed"]
    for key in ("before", "after"):
        parameters = execution["payload"]["before" if key == "before" else "target"]
        expected = simulate_quality(s.context, parameters, observation[key + "_seed"], sample_count)
        assert {name: observation[key][name] for name in expected} == expected
        assert observation[key]["sample_count"] == sample_count
        assert observation[key]["defect_count"] / sample_count == observation[key]["defect_rate"]
        low, high = observation[key]["rate_ci95"]
        assert 0 <= low <= observation[key]["defect_rate"] + 1e-12 <= high + 1e-12 <= 1 + 1e-12
    difference = observation["after"]["defect_rate"] - observation["before"]["defect_rate"]
    assert observation["difference_after_minus_before"] == difference
    low, high = observation["difference_ci95"]
    assert -1 <= low <= difference <= high <= 1
    expected_label = (
        "synthetic_observed_improvement"
        if high < 0
        else "worsened"
        if low > 0
        else "no_clear_improvement"
    )
    assert observation["conclusion"] == expected_label
    assert s.observe(sample_count=sample_count) == observed
    assert_error("idempotency_conflict", s.observe, sample_count=sample_count, seed=17)
    assert_error(
        "invalid_state",
        s.actions.observe,
        s.principal,
        s.recommendation["object_id"],
        action_id="reroll",
    )
    s.restart()
    assert s.get("recommendation") == observed


@pytest.mark.parametrize("change", ["device", "device_version", "receipt", "context"])
def test_observation_rechecks_live_receipt_device_and_source(scenario, change):
    s = scenario
    execution = s.readback()
    s.clock.value += 60_000
    if change.startswith("device"):
        s.mutate(
            "device",
            s.context.machine_id,
            lambda p: p["current"].update(pressure="32") if change == "device" else None,
        )
    elif change == "receipt":
        s.mutate(
            "device_receipt",
            execution["object_id"],
            lambda p: p.update(applied_at_ms=p["applied_at_ms"] + 1),
        )
    else:
        s.mutate("context", s.context.context_id, lambda p: p.update(revision=p["revision"] + 1))
    with pytest.raises(DomainError):
        s.observe()
    assert s.get("recommendation")["status"] == "executed"
    assert "observation" not in s.get("recommendation")["payload"]


def test_readback_and_observation_record_actual_past_apply_even_if_active_model_changes(scenario):
    s = scenario
    execution = s.apply(fault="after_apply")
    s.service.register_model(s.principal, s.model, s.context.plant_id, "new-model-after-apply")
    assert (
        s.actions.readback(s.principal, execution["object_id"], action_id="readback")["status"]
        == "executed"
    )
    s.clock.value += 60_000
    assert s.observe()["status"] == "observed"


def test_idempotency_binds_stage_body_and_target_before_any_mutation(scenario):
    s = scenario
    approved = s.approve()
    assert s.approve() == approved
    assert_error(
        "idempotency_conflict",
        s.actions.review,
        s.principal,
        approved["object_id"],
        action_id="review-action",
        recommendation_hash=s.recommendation["content_hash"],
        decision="approved",
        note="Changed note",
    )
    assert_error(
        "idempotency_conflict", s.request, recommendation=approved, action_id="review-action"
    )
    second = s.submit("synthetic-task-second")
    assert_error("idempotency_conflict", s.approve, recommendation=second)
    execution = s.request(recommendation=approved)
    assert s.request(recommendation=approved) == execution
    assert_error("invalid_state", s.request, recommendation=approved, action_id="different-key")
    applied = s.actions.apply_synthetic(s.principal, execution["object_id"], action_id="apply")
    assert (
        s.actions.apply_synthetic(s.principal, execution["object_id"], action_id="apply") == applied
    )
    done = s.actions.readback(s.principal, execution["object_id"], action_id="readback")
    assert s.actions.readback(s.principal, execution["object_id"], action_id="readback") == done
    assert s.get("device")["version"] == 2


@pytest.mark.parametrize("phase", ["review", "request", "apply", "readback", "observe"])
def test_every_phase_revalidates_acl_before_replaying(scenario, phase):
    s = scenario
    if phase == "review":
        s.approve()
        replay = s.approve
    elif phase == "request":
        approved = s.approve()
        s.request(recommendation=approved)

        def replay():
            return s.request(recommendation=approved)
    elif phase == "apply":
        execution = s.apply()

        def replay():
            return s.actions.apply_synthetic(
                s.principal, execution["object_id"], action_id="apply-action"
            )
    elif phase == "readback":
        execution = s.readback()

        def replay():
            return s.actions.readback(
                s.principal, execution["object_id"], action_id="readback-action"
            )
    else:
        s.readback()
        s.clock.value += 60_000
        s.observe()
        replay = s.observe
    s.store.set_user(s.principal.user_id, active=False)
    assert_error("forbidden", replay)


def test_owner_plant_and_review_role_are_not_supplied_by_the_caller(scenario):
    s = scenario
    other = s.store.create_user(
        "other-reviewer", "test-other-" + "b" * 40, "reviewer", [s.context.plant_id]
    )
    assert_error(
        "not_found",
        s.actions.review,
        other,
        s.recommendation["object_id"],
        action_id="steal",
        recommendation_hash=s.recommendation["content_hash"],
        decision="approved",
        note="no",
    )
    s.store.set_user(s.principal.user_id, role="operator")
    operator = s.store.authenticate(s.token)
    assert_error(
        "forbidden",
        s.actions.review,
        operator,
        s.recommendation["object_id"],
        action_id="review",
        recommendation_hash=s.recommendation["content_hash"],
        decision="approved",
        note="no",
    )
    s.store.set_user(s.principal.user_id, role="reviewer", plants=["different-plant"])
    current = s.store.authenticate(s.token)
    assert_error(
        "not_found",
        s.actions.review,
        current,
        s.recommendation["object_id"],
        action_id="review",
        recommendation_hash=s.recommendation["content_hash"],
        decision="approved",
        note="no",
    )


def test_operator_can_execute_owned_reviewed_synthetic_recommendation(scenario):
    s = scenario
    approved = s.approve()
    s.store.set_user(s.principal.user_id, role="operator")
    s.principal = s.store.authenticate(s.token)
    assert_error("forbidden", s.approve)
    execution = s.request(recommendation=approved)
    assert (
        s.actions.apply_synthetic(s.principal, execution["object_id"], action_id="apply")["status"]
        == "applied"
    )
    assert (
        s.actions.readback(s.principal, execution["object_id"], action_id="readback")["status"]
        == "executed"
    )
    s.clock.value += 60_000
    assert s.observe()["status"] == "observed"


def test_device_and_receipt_rollback_together_if_transaction_fails(scenario, monkeypatch):
    s = scenario
    execution = s.request()
    original = s.get("device")
    record_receipt = s.store.record_receipt

    def fail_at_record(conn, principal, action_id, request, response):
        if request["operation"] == "apply_synthetic":
            raise RuntimeError("injected local transaction failure")
        return record_receipt(conn, principal, action_id, request, response)

    monkeypatch.setattr(s.store, "record_receipt", fail_at_record)
    with pytest.raises(RuntimeError, match="transaction failure"):
        s.actions.apply_synthetic(
            s.principal, execution["object_id"], action_id="apply", fault="after_apply"
        )
    assert s.get("device") == original
    assert s.get("execution", execution["object_id"])["status"] == "pending"
    assert s.get("recommendation")["status"] == "execution_pending"
    assert s.store.list_objects("device_receipt", s.principal) == []
    s.restart()
    assert (
        s.actions.apply_synthetic(s.principal, execution["object_id"], action_id="apply")["status"]
        == "applied"
    )


@pytest.mark.parametrize("failure", ["crash", "revoke", "device_drift"])
def test_device_commit_survives_failure_before_independent_ack(scenario, monkeypatch, failure):
    s = scenario
    execution = s.request()
    identifier = execution["object_id"]
    original_ack = s.actions._acknowledge

    def interrupt(principal, execution_id, action_id, request, reservation):
        # A fresh Store can read the commit before the ACK phase even begins.
        independent = Store(s.root)
        device = independent.get_object("device", s.context.machine_id, principal)
        assert device["payload"]["current"] == execution["payload"]["target"]
        assert (
            independent.get_object("device_receipt", identifier, principal)["status"] == "applied"
        )
        assert independent.get_object("execution", identifier, principal)["status"] == "unknown"
        if failure == "crash":
            raise RuntimeError("process stopped before ACK")
        if failure == "revoke":
            independent.set_user(principal.user_id, active=False)
        else:
            s.mutate("device", s.context.machine_id, lambda p: p["current"].update(pressure="32"))
        return original_ack(principal, execution_id, action_id, request, reservation)

    monkeypatch.setattr(s.actions, "_acknowledge", interrupt)
    with pytest.raises(RuntimeError if failure == "crash" else DomainError):
        s.actions.apply_synthetic(s.principal, identifier, action_id="interrupted-apply")
    if failure == "revoke":
        s.store.set_user(s.principal.user_id, active=True)
    s.restart()
    assert s.get("execution", identifier)["status"] == "unknown"
    assert s.get("recommendation")["status"] == "unknown"
    device = s.get("device")
    replay = s.actions.apply_synthetic(s.principal, identifier, action_id="interrupted-apply")
    assert replay["status"] == "unknown"
    assert_error(
        "invalid_state", s.actions.apply_synthetic, s.principal, identifier, action_id="resend"
    )
    assert_error(
        "idempotency_conflict",
        s.actions.readback,
        s.principal,
        identifier,
        action_id="interrupted-apply",
    )
    assert s.get("device") == device
    result = s.actions.readback(s.principal, identifier, action_id="recover")
    assert result["status"] == ("unknown" if failure == "device_drift" else "executed")
    assert s.get("device") == device


def test_concurrent_readback_before_ack_is_not_overwritten(scenario, monkeypatch):
    s = scenario
    execution = s.request()
    original_ack = s.actions._acknowledge

    def read_before_ack(principal, execution_id, action_id, request, reservation):
        assert (
            s.actions.readback(principal, execution_id, action_id="early-readback")["status"]
            == "executed"
        )
        return original_ack(principal, execution_id, action_id, request, reservation)

    monkeypatch.setattr(s.actions, "_acknowledge", read_before_ack)
    result = s.actions.apply_synthetic(s.principal, execution["object_id"], action_id="apply")
    assert result["status"] == "unknown"  # conservative original ACK outcome
    assert s.get("execution", execution["object_id"])["status"] == "executed"
    assert s.get("recommendation")["status"] == "executed"
    assert s.get("device")["version"] == 2


def parallel_pair(function):
    barrier = Barrier(2)

    def run(index):
        barrier.wait(timeout=10)
        try:
            return function(index)
        except DomainError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(run, range(2)))


@pytest.mark.parametrize("same_key", [False, True])
def test_concurrent_execution_requests_have_one_device_request(scenario, same_key):
    s = scenario
    approved = s.approve()
    results = parallel_pair(
        lambda index: s.request(
            recommendation=approved, action_id="concurrent" if same_key else f"concurrent-{index}"
        )
    )
    successes = [result for result in results if isinstance(result, dict)]
    assert len(successes) == (2 if same_key else 1)
    if same_key:
        assert results[0] == results[1]
    else:
        assert [r.code for r in results if isinstance(r, DomainError)] == ["invalid_state"]
    assert len(s.store.list_objects("execution", s.principal)) == 1
    assert s.get("device")["version"] == 1


@pytest.mark.parametrize("same_key", [False, True])
def test_concurrent_apply_is_single_local_device_mutation(scenario, same_key):
    s = scenario
    execution = s.request()
    results = parallel_pair(
        lambda index: s.actions.apply_synthetic(
            s.principal,
            execution["object_id"],
            action_id="concurrent" if same_key else f"concurrent-{index}",
            fault="after_apply",
        )
    )
    successes = [result for result in results if isinstance(result, dict)]
    assert len(successes) == (2 if same_key else 1)
    assert all(result["status"] == "unknown" for result in successes)
    assert s.get("device")["version"] == 2
    assert s.get("device")["payload"]["revision"] == 2
    assert len(s.store.list_objects("device_receipt", s.principal)) == 1


def test_competing_recommendations_cannot_both_apply_old_device_revision(scenario):
    s = scenario
    second = s.submit("synthetic-task-second")
    first_execution = s.request()
    second_approved = s.approve(recommendation=second, action_id="review-second")
    second_execution = s.request(recommendation=second_approved, action_id="request-second")
    executions = [first_execution, second_execution]
    results = parallel_pair(
        lambda index: s.actions.apply_synthetic(
            s.principal, executions[index]["object_id"], action_id=f"apply-{index}"
        )
    )
    assert sum(isinstance(result, dict) for result in results) == 1
    assert s.get("device")["version"] == 2
    assert len(s.store.list_objects("device_receipt", s.principal)) == 1


@pytest.mark.parametrize("phase", ["review", "readback", "observe"])
@pytest.mark.parametrize("same_key", [False, True])
def test_concurrent_remaining_phases_use_cas_and_idempotency(scenario, phase, same_key):
    s = scenario
    if phase == "readback":
        execution = s.apply(fault="after_apply")
    elif phase == "observe":
        s.readback()
        s.clock.value += 60_000

    def invoke(index):
        action_id = "concurrent-phase" if same_key else f"concurrent-phase-{index}"
        if phase == "review":
            return s.approve(action_id=action_id)
        if phase == "readback":
            return s.actions.readback(s.principal, execution["object_id"], action_id=action_id)
        return s.actions.observe(s.principal, s.recommendation["object_id"], action_id=action_id)

    results = parallel_pair(invoke)
    successes = [result for result in results if isinstance(result, dict)]
    assert len(successes) == (2 if same_key else 1)
    if same_key:
        assert results[0] == results[1]
    else:
        assert [result.code for result in results if isinstance(result, DomainError)] == [
            "invalid_state"
        ]
    assert (
        s.get("recommendation")["status"]
        == {"review": "approved", "readback": "executed", "observe": "observed"}[phase]
    )
    assert s.get("device")["version"] == (1 if phase == "review" else 2)


@pytest.mark.parametrize("fault", [True, 1, "before_apply", "http://device", {}])
def test_adapter_does_not_accept_arbitrary_fault_or_endpoint(scenario, fault):
    s = scenario
    execution = s.request()
    assert_error(
        "invalid_fault",
        s.actions.apply_synthetic,
        s.principal,
        execution["object_id"],
        action_id="apply",
        fault=fault,
    )
    assert s.get("device")["version"] == 1


def test_invalid_runtime_clock_is_not_coerced(scenario):
    s = scenario
    s.clock.value = True
    assert_error("invalid_time", s.approve)
    assert s.get("recommendation")["status"] == "pending_review"
