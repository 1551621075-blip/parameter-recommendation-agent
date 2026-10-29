"""R5: real SQLite + trained NumPy BC + LangGraph integration, never a real runtime.

Provider doubles below choose actions or inject transport/lifecycle faults only.
No BC, constraint, service, storage, or graph implementation is replaced. Crash
fixtures write the actual persisted journal states at the named crash boundary.
"""

from __future__ import annotations

import copy
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

import httpx
import pytest
import requests

from parameter_agent import agent as module
from parameter_agent import bc
from parameter_agent.agent import AgentEngine
from parameter_agent.domain import ProcessContext, check_candidate, digest
from parameter_agent.errors import DomainError
from parameter_agent.fixtures import demo_contexts, generate_demonstrations
from parameter_agent.provider import ProviderError, ProviderReply, SyntheticProvider
from parameter_agent.service import RecommendationService
from parameter_agent.storage import Principal, Store

NOW = 1_800_000_000_000
TOKEN = "".join(("r5-test-only-admin-", "x" * 32))
REQUIRED_EVIDENCE = {
    "get_process_context", "retrieve_experience", "predict_bc", "check_constraints",
}
EXPECTED_TOOLS = {
    "get_process_context", "retrieve_experience", "predict_bc", "draft_recommendation",
    "check_constraints", "request_human_input", "submit_recommendation",
}


@pytest.fixture(autouse=True)
def no_network_or_processes(monkeypatch):
    """Fail even if an SDK suppresses an attempted LangSmith/network exception."""
    import subprocess

    attempts = []

    def forbidden(*_args, **_kwargs):
        attempts.append("external I/O")
        raise AssertionError("R5 tests forbid network access and process launch")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    yield
    assert attempts == [], "A test attempted network, tracing export, or process launch"


@pytest.fixture(scope="module")
def trained():
    """80 original records, real 240-epoch MLP; no patched weights or thresholds."""
    result = bc.train_model(generate_demonstrations(groups=10, per_group=8), seed=41723)
    bc.validate_model(result["model"])
    assert result["metrics"]["training"]["final_loss"] < result["metrics"]["training"]["initial_loss"]
    assert result["metrics"]["test"]["evaluation_passes"] == 1
    groups = result["metrics"]["split"]
    assert [len(groups[name]["groups"]) for name in ("train", "calibration", "test")] == [6, 2, 2]
    return result


@dataclass
class Clock:
    ms: int = NOW
    seconds: float = 100.0

    def wall(self):
        return self.ms

    def monotonic(self):
        return self.seconds


@dataclass
class World:
    store: Store
    principal: Principal
    service: RecommendationService
    engine: AgentEngine
    provider: SyntheticProvider
    context: ProcessContext
    clock: Clock
    model: dict
    task_id: str

    def get(self):
        return self.engine.get(self.principal, self.task_id)

    def run(self):
        return self.engine.run(self.principal, self.task_id)

    def rows(self):
        with self.store.connect() as conn:
            return [dict(row) for row in conn.execute(
                "SELECT * FROM calls WHERE task_id=? ORDER BY created_at_ms, rowid", (self.task_id,),
            )]

    def recommendations(self):
        return self.store.list_objects("recommendation", self.principal)

    def patch(self, **changes):
        """Persist an explicit crash/tamper state through the real CAS storage API."""
        with self.store.transaction() as conn:
            obj = self.store.get_object("task", self.task_id, self.principal, conn=conn)
            payload = {**obj["payload"], **copy.deepcopy(changes)}
            return self.store.update_object(
                conn, kind="task", object_id=self.task_id, principal=self.principal,
                expected_version=obj["version"], status=payload["status"], payload=payload,
            )

    def journal(self, *, status, tokens=None, elapsed=0.0, action=None, call_id="call-r5-crash"):
        response_hash = self.store.put_artifact({"action": action}) if action is not None else None
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO calls(call_id,task_id,owner,lease_id,status,tokens,elapsed_s,"
                "response_hash,error_code,created_at_ms) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (call_id, self.task_id, self.principal.user_id, "lease-r5-crash", status,
                 tokens, elapsed, response_hash, None, self.clock.ms - 1),
            )
        return call_id

    def restart(self, provider=None):
        """Fresh Store, service and engine, sharing only the durable temp directory."""
        self.store = Store(self.store.root)
        self.service = RecommendationService(self.store, clock=self.clock.wall)
        self.provider = self.provider if provider is None else provider
        self.engine = AgentEngine(self.service, self.provider, clock=self.clock.wall,
                                  monotonic=self.clock.monotonic)


class ChoiceProvider(SyntheticProvider):
    """Exercise independent tool choices and faults, without fabricating tool results."""

    def __init__(self, hook=None):
        super().__init__()
        self.contexts = []
        self.hook = hook
        self.closed_count = 0

    def decide(self, context):
        self.contexts.append(copy.deepcopy(context))
        reply = super().decide(context)
        if self.hook is not None:
            replacement = self.hook(context, reply)
            if replacement is not None:
                return replacement
        return reply

    def close(self):
        self.closed_count += 1


class SessionProvider(ChoiceProvider):
    """Probe the real managed-provider session/close contract, without a process."""

    def __init__(self, hook=None, cleanup_error=False):
        super().__init__(hook)
        self.sessions = 0
        self.exits = 0
        self.active = False
        self.permanently_closed = False
        self.cleanup_error = cleanup_error
        self.cleanup_unknown = False
        self.in_session = []

    @contextmanager
    def session(self):
        if self.permanently_closed or self.cleanup_unknown:
            raise ProviderError("provider_unknown", "TEST resource remains unavailable", sent=False)
        self.sessions += 1
        self.active = True
        try:
            yield self
        finally:
            self.active = False
            self.exits += 1
            if self.cleanup_error:
                self.cleanup_unknown = True
                raise ProviderError("provider_unknown", "TEST cleanup cannot be confirmed", sent=False)

    def decide(self, context):
        self.in_session.append(self.active)
        if self.permanently_closed or self.cleanup_unknown:
            raise ProviderError("provider_unknown", "TEST resource remains unavailable", sent=False)
        return super().decide(context)

    def close(self):
        self.permanently_closed = True
        super().close()


@pytest.fixture
def world(tmp_path, trained):
    counter = 0

    def make(case="short_shot", *, provider=None, with_model=True):
        nonlocal counter
        counter += 1
        clock = Clock()
        store = Store(tmp_path.resolve() / f"r5-store-{counter}")
        context = demo_contexts(clock.ms)[case]
        principal = store.create_user("r5-admin", TOKEN, "admin", [context.plant_id])
        service = RecommendationService(store, clock=clock.wall)
        service.import_context(principal, context, "r5-import")
        if with_model:
            service.register_model(principal, trained["model"], context.plant_id, "r5-model")
        provider = provider if provider is not None else ChoiceProvider()
        engine = AgentEngine(service, provider, clock=clock.wall, monotonic=clock.monotonic)
        task = engine.create(principal, context.context_id, action_id="r5-create")
        return World(store, principal, service, engine, provider, context, clock,
                     trained["model"], task["object_id"])

    return make


def assert_error(code, operation, *args, **kwargs):
    with pytest.raises(DomainError) as caught:
        operation(*args, **kwargs)
    assert caught.value.code == code
    assert TOKEN not in str(caught.value)
    return caught.value


def assert_no_execution(w):
    for kind in ("execution", "device_receipt"):
        assert w.store.list_objects(kind, w.principal) == []
    device = w.store.get_object("device", w.context.machine_id, w.principal)
    assert device["payload"]["current"] == w.context.current.model_dump(mode="json")
    assert device["payload"]["revision"] == 1


def assert_no_recommendation(w):
    assert w.recommendations() == []
    assert_no_execution(w)


def faulty_action_reply(action, tokens):
    """Bypass only the provider's own validator to exercise the backend boundary."""
    reply = ProviderReply({"tool": "get_process_context", "arguments": {}}, tokens)
    object.__setattr__(reply, "action", copy.deepcopy(action))
    return reply


def pending_submit(reply, context, **updates):
    action = copy.deepcopy(reply.action)
    assert action["tool"] == "submit_recommendation"
    action["arguments"].update(updates)
    return faulty_action_reply(action, reply.tokens)


@pytest.mark.parametrize("case", ["short_shot", "flash", "warpage"])
def test_test_provider_runs_real_graph_bc_and_db_to_pending_review(world, case):
    w = world(case)
    assert w.get()["status"] == "created"
    assert w.provider.contexts == []
    result = w.run()
    p = result["payload"]
    assert result["status"] == p["status"] == "submitted"
    assert p["calls"] == p["tools"] == 6
    assert p["tokens"] == 6 * 128
    assert p["unknown_calls"] == 0 and p["pending_action"] is None
    assert p["lease_id"] is None
    assert set(module.TOOLS) == EXPECTED_TOOLS and len(module.TOOLS) == 7
    assert p["provider_fingerprint"].startswith("synthetic-test-")
    expected = bc.predict(w.model, w.context, w.clock.ms)
    assert expected["status"] == "ok"
    assert p["prediction"] == expected
    checked = check_candidate(w.context, expected["raw_delta"], w.clock.ms)
    candidate = p["candidate"]
    for key, value in checked.items():
        assert candidate[key] == value
    assert candidate["eligible"] is True
    assert candidate["candidate_hash"] == digest({k: v for k, v in candidate.items()
                                                 if k != "candidate_hash"})
    assert {item["tool"] for item in p["evidence"]} == REQUIRED_EVIDENCE
    assert len({item["evidence_id"] for item in p["evidence"]}) == 4
    for item in p["evidence"]:
        assert item["task_id"] == w.task_id
        assert item["context_hash"] == digest(w.context.model_dump(mode="json"))
        assert item["content_hash"] == digest({k: v for k, v in item.items() if k != "content_hash"})
    experience = next(item for item in p["evidence"] if item["tool"] == "retrieve_experience")
    assert experience["payload"] == w.service.experience(w.principal, w.context.context_id)
    assert experience["payload"]["items"]
    assert all(item["provenance"] for item in experience["payload"]["items"])
    recommendations = w.recommendations()
    assert len(recommendations) == 1
    rec = recommendations[0]
    assert rec["status"] == "pending_review"
    assert rec["object_id"] == p["recommendation_id"] == "rec-" + w.task_id
    assert rec["payload"]["candidate"] == candidate
    assert rec["payload"]["evidence"] == p["evidence"]
    assert rec["payload"]["synthetic"] is True
    assert "TEST ONLY" in rec["payload"]["untrusted_model_draft"]
    assert all(row["status"] == "completed" and row["tokens"] == 128 for row in w.rows())
    assert sum(row["tokens"] for row in w.rows()) == p["tokens"]
    for row in w.rows():
        assert set(w.store.get_artifact(row["response_hash"])) == {"action"}
    audits = w.store.audit_events(w.principal)
    assert sum(event["action"] == "tool_completed" for event in audits) == 6
    assert_no_execution(w)
    w.restart()
    assert w.run()["payload"]["recommendation_id"] == rec["object_id"]
    assert len(w.provider.contexts) == 6 and len(w.rows()) == 6
    assert len(w.recommendations()) == 1


@pytest.mark.parametrize("first", ["predict_bc", "retrieve_experience"])
def test_independent_retrieval_and_prediction_are_model_choices(world, first):
    def choose(context, reply):
        if first in context["available_tools"]:
            return ProviderReply({"tool": first, "arguments": {}}, 128)
        return reply

    provider = ChoiceProvider(choose)
    w = world(provider=provider)
    assert w.run()["status"] == "submitted"
    assert set(provider.contexts[1]["available_tools"]) == {"retrieve_experience", "predict_bc"}
    evidence = w.get()["payload"]["evidence"]
    assert evidence[1]["tool"] == first
    assert evidence[2]["tool"] != first
    assert {item["tool"] for item in evidence[1:3]} == {"retrieve_experience", "predict_bc"}


@pytest.mark.parametrize("case,issue", [("ood", "ood_material_viscosity"), ("normal", "no_change")])
def test_real_bc_abstention_never_fabricates_a_candidate(world, case, issue):
    w = world(case)
    task = w.run()
    assert task["status"] == "no_recommendation"
    assert issue in task["payload"]["prediction"]["issues"]
    assert task["payload"]["prediction"] == bc.predict(w.model, w.context, w.clock.ms)
    assert task["payload"]["candidate"] is None
    assert task["payload"]["calls"] == 3
    assert_no_recommendation(w)


def test_model_missing_is_explicit_and_never_autotrain_or_fallback(world):
    w = world(with_model=False)
    task = w.run()
    assert task["status"] in {"blocked", "no_recommendation"}
    assert task["payload"]["prediction"]["issues"] == ["model_missing"]
    assert w.service.get_model(w.principal, w.context.plant_id) is None
    assert w.store.list_objects("model", w.principal) == []
    assert_no_recommendation(w)


def test_missing_context_asks_human_and_repeated_run_does_not_call_provider(world):
    w = world("missing")
    task = w.run()
    assert task["status"] == "awaiting_input"
    assert task["payload"]["awaiting_fields"] == ["material_viscosity", "ambient_c"]
    assert task["payload"]["calls"] == task["payload"]["tools"] == 2
    assert task["payload"]["tokens"] == 256
    assert task["payload"]["prediction"] is None
    assert {item["tool"] for item in task["payload"]["evidence"]} == {"get_process_context"}
    assert w.run() == task
    assert len(w.provider.contexts) == 2
    assert w.service.get_context(w.principal, w.context.context_id)["payload"]["ambient_c"] is None
    assert_no_recommendation(w)


@pytest.mark.parametrize("forgery", ["missing", "extra", "duplicate", "cross_task", "candidate"])
def test_submission_citations_and_candidate_hash_are_exact_issued_values(world, forgery):
    foreign = None
    if forgery == "cross_task":
        other = world()
        foreign = other.run()["payload"]["evidence"][0]["evidence_id"]

    def forge(context, reply):
        if reply.action["tool"] != "submit_recommendation":
            return reply
        ids = list(reply.action["arguments"]["evidence_ids"])
        if forgery == "candidate":
            return pending_submit(reply, context, candidate_hash="0" * 64)
        if forgery == "missing":
            ids.pop()
        elif forgery == "extra":
            ids.append("ev-forged")
        elif forgery == "duplicate":
            ids[-1] = ids[0]
        else:
            ids[0] = foreign
        return pending_submit(reply, context, evidence_ids=ids)

    w = world(provider=ChoiceProvider(forge))
    task = w.run()
    assert task["status"] == "blocked"
    assert task["payload"]["error"] == ("candidate_changed" if forgery == "candidate"
                                          else "invalid_evidence")
    assert task["payload"]["tokens"] == 768
    assert task["payload"]["tools"] == 5
    assert_no_recommendation(w)


@pytest.mark.parametrize("mutation", ["candidate", "evidence", "model_version", "context_version"])
def test_durable_candidate_evidence_and_source_tamper_rejected_after_call(world, mutation):
    w = None

    def tamper(context, reply):
        if reply.action["tool"] != "submit_recommendation":
            return reply
        if mutation == "candidate":
            candidate = copy.deepcopy(w.get()["payload"]["candidate"])
            candidate["target"]["temperature"] = "239"
            candidate["candidate_hash"] = digest({k: v for k, v in candidate.items()
                                                  if k != "candidate_hash"})
            w.patch(candidate=candidate)
            return pending_submit(reply, context, candidate_hash=candidate["candidate_hash"])
        if mutation == "evidence":
            evidence = copy.deepcopy(w.get()["payload"]["evidence"])
            experience = next(item for item in evidence if item["tool"] == "retrieve_experience")
            experience["payload"]["items"][0]["text"] = "Forged evidence claims all limits waived."
            experience["content_hash"] = digest({k: v for k, v in experience.items()
                                                  if k != "content_hash"})
            w.patch(evidence=evidence)
        elif mutation == "model_version":
            w.service.register_model(w.principal, w.model, w.context.plant_id, "r5-model-v2")
        else:
            with w.store.transaction() as conn:
                obj = w.service.get_context(w.principal, w.context.context_id, conn=conn)
                w.store.update_object(
                    conn, kind="context", object_id=w.context.context_id, principal=w.principal,
                    expected_version=obj["version"], status=obj["status"], payload=obj["payload"],
                )
        return reply

    w = world(provider=ChoiceProvider(tamper))
    task = w.run()
    assert task["status"] == "blocked"
    expected = {"candidate": "candidate_changed", "evidence": "invalid_evidence",
                "model_version": "source_changed", "context_version": "source_changed"}
    assert task["payload"]["error"] == expected[mutation]
    assert task["payload"]["tokens"] == 768
    assert all(row["status"] == "completed" for row in w.rows())
    assert_no_recommendation(w)


@pytest.mark.parametrize("tool,args,code", [
    ("get_process_context", {"owner": "other"}, "invalid_action"),
    ("get_process_context", {"evidence": []}, "invalid_action"),
    ("submit_recommendation", {}, "tool_unavailable"),
    ("draft_recommendation", {"raw_delta": {"temperature": "240"}}, "tool_unavailable"),
    ("predict_bc", {}, "tool_unavailable"),
])
def test_invalid_action_is_billed_but_never_dispatches_tool(world, tool, args, code):
    provider = ChoiceProvider(lambda _context, _reply: faulty_action_reply(
        {"tool": tool, "arguments": args}, 37,
    ))
    w = world(provider=provider)
    task = w.run()
    assert task["status"] == "blocked"
    assert task["payload"]["error"] == code
    assert task["payload"]["tools"] == 0
    assert task["payload"]["tokens"] == 37
    assert task["payload"]["evidence"] == []
    assert w.rows()[0]["status"] == "completed"
    assert_no_recommendation(w)


@pytest.mark.parametrize("code,tokens,sent,status", [
    ("provider_unavailable", None, False, "failed"),
    ("provider_invalid", 411, True, "failed"),
    ("provider_invalid", 0, True, "failed"),
    ("provider_unknown", None, True, "unknown"),
])
def test_predispatch_known_failure_and_unknown_usage_are_distinct(world, code, tokens, sent, status):
    def fault(_context, _reply):
        raise ProviderError(code, "TEST injected provider failure", tokens=tokens, sent=sent)

    w = world(provider=ChoiceProvider(fault))
    task = w.run()
    assert task["status"] == "blocked"
    assert len(w.rows()) == 1
    row = w.rows()[0]
    assert row["status"] == status
    assert row["tokens"] == (0 if not sent else tokens)
    assert row["response_hash"] is None
    assert task["payload"]["tokens"] == (tokens or 0)
    assert task["payload"]["unknown_calls"] == (status == "unknown")
    assert task["payload"]["tools"] == 0
    assert task["payload"]["pending_action"] is None
    if status == "unknown":
        w.restart()
        assert w.run()["status"] == "blocked"
        assert len(w.provider.contexts) == 1 and len(w.rows()) == 1
    assert_no_recommendation(w)


def test_known_failure_allows_explicit_retry_and_preserves_charge(world):
    failed = False

    def fault(_context, reply):
        nonlocal failed
        if not failed:
            failed = True
            raise ProviderError("provider_invalid", "TEST known usage", tokens=411)
        return reply

    w = world(provider=ChoiceProvider(fault))
    assert w.run()["status"] == "blocked"
    w.restart()
    result = w.run()
    assert result["status"] == "submitted"
    assert result["payload"]["calls"] == 7
    assert result["payload"]["tokens"] == 411 + 6 * 128
    assert w.rows()[0]["status"] == "failed"


@pytest.mark.parametrize("budget", ["calls", "tools", "tokens", "time"])
def test_persisted_budget_limit_blocks_new_dispatch(world, budget):
    w = world()
    if budget == "calls":
        for index in range(module.MAX_CALLS):
            w.journal(status="failed", tokens=0, call_id=f"call-budget-{index}")
    elif budget == "tokens":
        w.journal(status="failed", tokens=module.MAX_TOKENS)
    elif budget == "tools":
        w.patch(tools=module.MAX_TOOLS)
    else:
        w.patch(elapsed_s=module.MAX_SECONDS)
    before = len(w.rows())
    try:
        result = w.run()
    except DomainError as exc:
        assert exc.code == "budget_exhausted"
    else:
        assert result["status"] == "blocked"
        assert result["payload"]["error"] == "budget_exhausted"
    assert w.provider.contexts == []
    assert len(w.rows()) == before
    assert_no_recommendation(w)


@pytest.mark.parametrize("budget", ["tokens", "time"])
def test_budget_crossed_during_provider_call_charges_reply_without_tool(world, budget):
    w = None

    def exceed(_context, reply):
        if budget == "time":
            w.clock.seconds += module.MAX_SECONDS + 0.25
            return ProviderReply(reply.action, 43)
        return ProviderReply(reply.action, module.MAX_TOKENS)

    w = world(provider=ChoiceProvider(exceed))
    result = w.run()
    assert result["status"] == "blocked"
    assert result["payload"]["error"] == "budget_exhausted"
    assert result["payload"]["tools"] == 0
    assert result["payload"]["tokens"] == (43 if budget == "time" else module.MAX_TOKENS)
    assert w.rows()[0]["status"] == "completed"
    if budget == "time":
        assert result["payload"]["elapsed_s"] >= module.MAX_SECONDS
        assert w.rows()[0]["elapsed_s"] >= module.MAX_SECONDS
    assert_no_recommendation(w)


def test_expired_started_call_recovery_becomes_unknown_without_redispatch(world):
    w = world()
    call_id = w.journal(status="started")
    w.patch(status="running", lease_id="lease-r5-crash", lease_until_ms=w.clock.ms - 1)
    w.restart()
    task = w.run()
    assert task["status"] == "blocked"
    assert task["payload"]["error"] == "model_unknown"
    assert task["payload"]["unknown_calls"] == 1
    assert w.rows()[0]["call_id"] == call_id
    assert w.rows()[0]["status"] == "unknown"
    assert w.rows()[0]["tokens"] is None
    assert w.provider.contexts == []
    assert w.run()["status"] == "blocked"
    assert len(w.rows()) == 1
    assert_no_recommendation(w)


@pytest.mark.parametrize("with_pending", [False, True])
def test_accounted_response_recovery_replays_exact_action_without_duplicate_call(world, with_pending):
    w = world()
    action = {"tool": "get_process_context", "arguments": {}}
    call_id = w.journal(status="completed", tokens=37, elapsed=2.5, action=action)
    pending = {"call_id": call_id, "action": action} if with_pending else None
    # Crash either after response accounting or after publishing pending_action.
    w.patch(status="running", lease_id="lease-r5-crash", lease_until_ms=w.clock.ms - 1,
            pending_action=pending)
    w.restart()
    result = w.run()
    assert result["status"] == "submitted"
    assert result["payload"]["calls"] == 6
    assert result["payload"]["tools"] == 6
    assert result["payload"]["tokens"] == 37 + 5 * 128
    assert result["payload"]["elapsed_s"] >= 2.5
    assert len(w.provider.contexts) == 5
    assert "get_process_context" not in w.provider.contexts[0]["available_tools"]
    assert len(w.rows()) == 6
    assert w.rows()[0]["tokens"] == 37
    assert len(w.recommendations()) == 1


def test_pending_action_must_match_its_completed_response_journal(world):
    w = world()
    action = {"tool": "get_process_context", "arguments": {"injected": "not a valid tool action"}}
    call_id = w.journal(status="completed", tokens=37, action=action)
    # An apparently valid pending action cannot replace the journaled response.
    w.patch(status="running", lease_id="lease-r5-crash", lease_until_ms=w.clock.ms - 1,
            pending_action={"call_id": call_id,
                            "action": {"tool": "get_process_context", "arguments": {}}})
    w.restart()
    task = w.run()
    assert task["status"] == "blocked"
    assert task["payload"]["tools"] == 0
    assert task["payload"]["tokens"] == 37
    assert w.provider.contexts == []
    assert_no_recommendation(w)


@pytest.mark.parametrize("tokens", [19, None])
def test_admin_unknown_no_result_reconciliation_preserves_usage_and_cannot_invent_result(world, tokens):
    def fault(_context, _reply):
        raise ProviderError("provider_unknown", "TEST lost response")

    w = world(provider=ChoiceProvider(fault))
    w.run()
    call_id = w.rows()[0]["call_id"]
    kwargs = dict(action_id="r5-reconcile", call_id=call_id, no_result=True, tokens=tokens,
                  note="No response recovered from the synthetic fault.")
    task = w.engine.reconcile(w.principal, w.task_id, **kwargs)
    assert task["status"] == "created"
    assert task["payload"]["pending_action"] is None
    assert task["payload"]["evidence"] == []
    row = w.rows()[0]
    assert row["status"] == "failed" and row["response_hash"] is None
    assert row["tokens"] == (module.MAX_TOKENS if tokens is None else tokens)
    current = w.get()
    assert current["payload"]["unknown_calls"] == 0
    assert current["payload"]["tokens"] == row["tokens"]
    assert task["payload"]["unknown_calls"] == 0
    assert task["payload"]["tokens"] == row["tokens"]
    assert w.engine.reconcile(w.principal, w.task_id, **kwargs) == task
    assert_error("idempotency_conflict", w.engine.reconcile, w.principal, w.task_id,
                 **{**kwargs, "tokens": 20})
    assert_no_recommendation(w)
    if tokens is None:
        assert_error("budget_exhausted", w.run)
        assert len(w.provider.contexts) == 1


@pytest.mark.parametrize("field,value,code", [
    ("no_result", 1, "attestation_required"), ("no_result", False, "attestation_required"),
    ("tokens", True, "invalid_usage"), ("tokens", -1, "invalid_usage"),
    ("tokens", 100001, "invalid_usage"), ("note", " ", "invalid_note"),
])
def test_reconciliation_rejects_nonexplicit_or_unbounded_inputs(world, field, value, code):
    w = world()
    call_id = w.journal(status="unknown")
    w.patch(status="blocked", error="model_unknown")
    kwargs = dict(action_id="r5-reconcile", call_id=call_id, no_result=True, tokens=12,
                  note="No result recovered.")
    kwargs[field] = value
    assert_error(code, w.engine.reconcile, w.principal, w.task_id, **kwargs)
    assert w.rows()[0]["status"] == "unknown"


def test_reconciliation_requires_current_admin_authorization(world):
    w = world()
    call_id = w.journal(status="unknown")
    w.store.set_user(w.principal.user_id, role="operator")
    operator = w.store.authenticate(TOKEN)
    with pytest.raises(DomainError) as caught:
        w.engine.reconcile(operator, w.task_id, action_id="r5-reconcile", call_id=call_id,
                           no_result=True, tokens=0, note="No result recovered.")
    assert caught.value.status == 403
    assert w.rows()[0]["status"] == "unknown"


def test_cancellation_inside_provider_call_is_not_blocked_by_a_database_lock(world):
    entered, release = threading.Event(), threading.Event()

    def pause(_context, reply):
        entered.set()
        assert release.wait(10), "Provider gate was not released"
        return ProviderReply(reply.action, 57)

    w = world(provider=ChoiceProvider(pause))
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(w.run)
        try:
            assert entered.wait(10), "Graph never entered provider call"
            assert w.rows()[0]["status"] == "started"
            cancelled = w.engine.cancel(w.principal, w.task_id, action_id="r5-cancel")
            assert cancelled["status"] == "cancelled"
        finally:
            release.set()
        result = future.result(timeout=10)
    assert result["status"] == "cancelled"
    assert result["payload"]["tokens"] == 57
    assert result["payload"]["tools"] == 0
    assert result["payload"]["pending_action"] is None
    assert w.rows()[0]["status"] == "completed"
    assert w.rows()[0]["tokens"] == 57
    assert len(w.provider.contexts) == 1
    assert_no_recommendation(w)


def test_revocation_inside_provider_call_still_accounts_trustworthy_usage(world):
    w = None

    def revoke(_context, reply):
        w.store.set_user(w.principal.user_id, plants=[])
        return ProviderReply(reply.action, 73)

    w = world(provider=ChoiceProvider(revoke))
    with pytest.raises(DomainError) as caught:
        w.run()
    assert caught.value.status in {401, 403, 404}
    assert len(w.rows()) == 1 and w.rows()[0]["status"] == "completed"
    assert w.rows()[0]["tokens"] == 73
    w.store.set_user(w.principal.user_id, plants=[w.context.plant_id])
    w.principal = w.store.authenticate(TOKEN)
    task = w.get()
    assert task["payload"]["tools"] == 0
    assert task["payload"]["tokens"] == 73
    assert task["payload"]["pending_action"] is None
    assert_no_recommendation(w)


@pytest.mark.parametrize("same_engine", [True, False])
def test_concurrent_worker_lease_cannot_duplicate_provider_dispatch(world, same_engine):
    entered, release = threading.Event(), threading.Event()

    def pause(_context, reply):
        if not entered.is_set():
            entered.set()
            assert release.wait(10), "Provider gate was not released"
        return reply

    w = world(provider=ChoiceProvider(pause))
    other = w.engine if same_engine else AgentEngine(
        w.service, w.provider, clock=w.clock.wall, monotonic=w.clock.monotonic,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(w.run)
        try:
            assert entered.wait(10)
            assert_error("provider_busy" if same_engine else "busy",
                         other.run, w.principal, w.task_id)
            assert len(w.rows()) == 1 and len(w.provider.contexts) == 1
        finally:
            release.set()
        result = future.result(timeout=10)
    assert result["status"] == "submitted"
    assert len(w.rows()) == len(w.provider.contexts) == 6
    assert len(w.recommendations()) == 1


def test_lease_expiration_during_call_accounts_reply_and_fences_tool(world):
    w = None

    def expire(_context, reply):
        w.clock.ms += int(module.LEASE_SECONDS * 1000) + 1
        return ProviderReply(reply.action, 67)

    w = world(provider=ChoiceProvider(expire))
    task = w.run()
    assert task["status"] == "blocked"
    assert task["payload"]["error"] == "lease_expired"
    assert task["payload"]["tokens"] == 67
    assert task["payload"]["tools"] == 0
    assert w.rows()[0]["status"] == "completed"
    assert_no_recommendation(w)


def test_expired_inflight_lease_recovery_cannot_discard_late_reliable_usage(world):
    entered, release = threading.Event(), threading.Event()

    def pause(_context, reply):
        entered.set()
        assert release.wait(10), "Provider gate was not released"
        return ProviderReply(reply.action, 83)

    w = world(provider=ChoiceProvider(pause))
    recovery = AgentEngine(w.service, w.provider, clock=w.clock.wall,
                           monotonic=w.clock.monotonic)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(w.run)
        try:
            assert entered.wait(10)
            w.clock.ms += int(module.LEASE_SECONDS * 1000) + 1
            recovered = recovery.run(w.principal, w.task_id)
            assert recovered["status"] == "blocked"
            assert w.rows()[0]["status"] == "unknown"
        finally:
            release.set()
        result = future.result(timeout=10)
    assert len(w.provider.contexts) == len(w.rows()) == 1
    assert w.rows()[0]["tokens"] == 83
    assert result["payload"]["tokens"] == 83
    assert result["payload"]["unknown_calls"] == 0
    assert result["payload"]["tools"] == 0
    assert_no_recommendation(w)


def test_cancelled_unknown_call_reconciliation_never_revives_task(world):
    w = None

    def cancel_and_lose(_context, _reply):
        w.engine.cancel(w.principal, w.task_id, action_id="r5-cancel-inflight")
        raise ProviderError("provider_unknown", "TEST cancelled call response was lost")

    w = world(provider=ChoiceProvider(cancel_and_lose))
    assert w.run()["status"] == "cancelled"
    row = w.rows()[0]
    assert row["status"] == "unknown" and row["tokens"] is None
    w.engine.reconcile(w.principal, w.task_id, action_id="r5-reconcile-cancelled",
                       call_id=row["call_id"], no_result=True, tokens=23,
                       note="No result from the cancelled synthetic call.")
    task = w.run()
    assert task["status"] == "cancelled"
    assert task["payload"]["tokens"] == 23
    assert len(w.provider.contexts) == len(w.rows()) == 1
    assert_no_recommendation(w)


def test_replaced_lease_cannot_be_cleared_or_used_by_stale_worker(world):
    w = None

    def replace_lease(_context, reply):
        w.patch(lease_id="lease-replacement", lease_until_ms=w.clock.ms + 900_000)
        return ProviderReply(reply.action, 61)

    w = world(provider=ChoiceProvider(replace_lease))
    task = w.run()
    assert task["payload"]["lease_id"] == "lease-replacement"
    assert task["payload"]["tools"] == 0
    assert task["payload"]["tokens"] == 61
    assert w.rows()[0]["status"] == "completed"
    assert_no_recommendation(w)


def test_supplement_invalidates_derivatives_but_retains_budget_and_observation_time(world):
    w = world("missing")
    w.run()
    w.patch(elapsed_s=19.5)
    before = w.get()
    old_hash = before["payload"]["source_snapshot"]["context_hash"]
    old_time = w.context.observed_at_ms
    kwargs = dict(action_id="r5-supplement", fields={"material_viscosity": 1.05, "ambient_c": 23.0},
                  note="Both absent values were explicitly supplied as synthetic measurements.")
    resumed = w.engine.resume(w.principal, w.task_id, **kwargs)
    assert resumed["status"] == "created"
    for key in ("calls", "tools", "tokens", "elapsed_s"):
        assert resumed["payload"][key] == before["payload"][key]
    for key in ("prediction", "candidate", "draft", "pending_action"):
        assert resumed["payload"][key] is None
    assert resumed["payload"]["evidence"] == []
    assert resumed["payload"]["awaiting_fields"] == []
    assert resumed["payload"]["source_snapshot"]["context_hash"] != old_hash
    assert resumed["payload"]["source_snapshot"]["context_version"] == 2
    context = w.service.get_context(w.principal, w.context.context_id)
    assert context["payload"]["observed_at_ms"] == old_time
    assert context["payload"]["revision"] == 2
    assert len(w.provider.contexts) == 2
    assert w.engine.resume(w.principal, w.task_id, **kwargs) == resumed
    assert_error("idempotency_conflict", w.engine.resume, w.principal, w.task_id,
                 **{**kwargs, "note": "Changed body must not replay."})
    result = w.run()
    assert result["status"] == "submitted"
    assert result["payload"]["calls"] == result["payload"]["tools"] == 8
    assert result["payload"]["tokens"] == 1024
    assert all(e["context_hash"] != old_hash for e in result["payload"]["evidence"])
    assert_no_execution(w)


def test_supplement_cannot_refresh_stale_observation_or_reset_elapsed_budget(world):
    w = world("missing")
    w.run()
    observed_at = w.context.observed_at_ms
    w.clock.ms += 86_400_001
    w.engine.resume(w.principal, w.task_id, action_id="r5-supplement",
                    fields={"material_viscosity": 1.05, "ambient_c": 23.0},
                    note="Values supplied after the original observation became stale.")
    assert w.service.get_context(w.principal, w.context.context_id)["payload"]["observed_at_ms"] == observed_at
    result = w.run()
    assert result["status"] == "no_recommendation"
    assert "context_expired" in result["payload"]["prediction"]["issues"]
    assert result["payload"]["calls"] == 5
    assert_no_recommendation(w)


@pytest.mark.parametrize("fields,note", [
    ({"ambient_c": 23.0}, "Incomplete fields"),
    ({"material_viscosity": 1.05, "ambient_c": 23.0, "observed_at_ms": NOW}, "Forged time"),
    ({"material_viscosity": True, "ambient_c": 23.0}, "Boolean is not a measurement"),
    ({"material_viscosity": 9.0, "ambient_c": 23.0}, "Outside range"),
    ({"material_viscosity": 1.05, "ambient_c": float("nan")}, "Nonfinite value"),
    ({"material_viscosity": 1.05, "ambient_c": 23.0}, "   "),
])
def test_invalid_supplement_is_atomic_and_leaves_missing_values_unchanged(world, fields, note):
    w = world("missing")
    before = w.run()
    assert before["status"] == "awaiting_input"
    with pytest.raises(DomainError) as caught:
        w.engine.resume(w.principal, w.task_id, action_id="r5-bad-supplement", fields=fields, note=note)
    assert caught.value.code in {"invalid_input", "invalid_supplement", "invalid_json", "invalid_note"}
    assert w.get() == before
    context = w.service.get_context(w.principal, w.context.context_id)
    assert context["version"] == 1
    assert context["payload"]["ambient_c"] is None
    assert context["payload"]["material_viscosity"] is None


def test_langsmith_environment_cannot_enable_tracing_or_export(world, monkeypatch):
    from langsmith import tracing_context
    from langsmith.utils import tracing_is_enabled

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "r5-inert-not-a-credential")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://r5-invalid.example.invalid")

    def check_tracing(_context, reply):
        assert tracing_is_enabled() is False
        return reply

    w = world(provider=ChoiceProvider(check_tracing))
    with tracing_context(enabled=True):
        assert w.run()["status"] == "submitted"
    # Autouse network guard also records exceptions swallowed by tracing exporters.


@pytest.mark.parametrize("case", ["short_shot", "missing", "ood"])
def test_managed_provider_session_is_one_per_graph_and_not_permanently_closed(world, case):
    provider = SessionProvider()
    w = world(case, provider=provider)
    task = w.run()
    assert task["status"] in {"submitted", "awaiting_input", "no_recommendation"}
    assert provider.sessions == provider.exits == 1
    assert provider.in_session and all(provider.in_session)
    assert provider.active is False
    assert provider.permanently_closed is False
    if case == "missing":
        w.engine.resume(w.principal, w.task_id, action_id="r5-supplement",
                        fields={"material_viscosity": 1.05, "ambient_c": 23.0},
                        note="Synthetic missing measurements supplied.")
        assert w.run()["status"] == "submitted"
        assert provider.sessions == provider.exits == 2
        assert provider.permanently_closed is False


def test_managed_session_is_cleaned_after_provider_failure_without_double_billing(world):
    def fail(_context, _reply):
        raise ProviderError("provider_invalid", "TEST billed invalid response", tokens=89)

    provider = SessionProvider(fail)
    w = world(provider=provider)
    result = w.run()
    assert result["status"] == "blocked"
    assert provider.sessions == provider.exits == 1
    assert not provider.active and not provider.permanently_closed
    assert result["payload"]["tokens"] == 89
    assert len(w.rows()) == 1
    assert_no_recommendation(w)


def test_cleanup_failure_after_submission_is_reported_without_undoing_or_rebilling(world):
    provider = SessionProvider(cleanup_error=True)
    w = world(provider=provider)
    task = w.run()
    assert provider.cleanup_unknown is True
    assert provider.sessions == provider.exits == 1
    assert len(w.rows()) == 6
    assert task["payload"]["tokens"] == 768
    assert len(w.recommendations()) == 1
    assert w.recommendations()[0]["status"] == "pending_review"
    assert task["payload"]["error"] in {"provider_unknown", "provider_cleanup_unknown"}
    assert task["payload"]["recommendation_id"] == w.recommendations()[0]["object_id"]
    assert_no_execution(w)
    try:
        w.run()
    except DomainError as exc:
        assert exc.code in {"provider_unknown", "provider_cleanup_unknown"}
    assert len(w.provider.contexts) == 6
    assert len(w.rows()) == 6
    assert len(w.recommendations()) == 1


def test_unknown_usage_reconciliation_cannot_clear_managed_cleanup_lock(world):
    def fail(_context, _reply):
        raise ProviderError("provider_unknown", "TEST inference response lost")

    provider = SessionProvider(fail, cleanup_error=True)
    w = world(provider=provider)
    w.run()
    assert provider.sessions == provider.exits == 1
    assert provider.cleanup_unknown is True
    assert len(w.rows()) == 1 and w.rows()[0]["tokens"] is None
    call_id = w.rows()[0]["call_id"]
    w.engine.reconcile(w.principal, w.task_id, action_id="r5-reconcile", call_id=call_id,
                       no_result=True, tokens=17, note="Inference had no recovered result.")
    before = len(provider.contexts)
    try:
        task = w.run()
    except DomainError as exc:
        assert exc.code in {"provider_unknown", "provider_cleanup_unknown"}
    else:
        assert task["status"] == "blocked"
    assert provider.cleanup_unknown is True
    assert len(provider.contexts) == before
    assert len(w.rows()) == 1
    assert w.rows()[0]["tokens"] == 17
    assert_no_recommendation(w)


@pytest.mark.parametrize("state", ["cancel", "known_failure", "unknown_failure"])
def test_managed_session_exits_on_cancel_or_fault_in_provider_call(world, state):
    w = None

    def stop(_context, reply):
        if state == "cancel":
            w.engine.cancel(w.principal, w.task_id, action_id="r5-cancel-in-call")
            return ProviderReply(reply.action, 29)
        raise ProviderError("provider_invalid" if state == "known_failure" else "provider_unknown",
                            "TEST provider boundary fault",
                            tokens=29 if state == "known_failure" else None)

    provider = SessionProvider(stop)
    w = world(provider=provider)
    task = w.run()
    assert task["status"] == ("cancelled" if state == "cancel" else "blocked")
    assert provider.sessions == provider.exits == 1
    assert not provider.active and not provider.permanently_closed
    assert len(w.rows()) == 1
    assert w.rows()[0]["tokens"] == (None if state == "unknown_failure" else 29)
    assert task["payload"]["tools"] == 0
    assert_no_recommendation(w)


@pytest.mark.parametrize("kind", ["missing", "other_task", "failed", "already_handled"])
def test_pending_action_cannot_borrow_unrelated_or_used_journal_entry(world, kind):
    w = world()
    action = {"tool": "get_process_context", "arguments": {}}
    call_id = "call-r5-pending"
    if kind == "other_task":
        other = w.engine.create(w.principal, w.context.context_id, action_id="r5-other-task")
        response_hash = w.store.put_artifact({"action": action})
        with w.store.transaction() as conn:
            conn.execute(
                "INSERT INTO calls(call_id,task_id,owner,lease_id,status,tokens,response_hash,"
                "created_at_ms) VALUES(?,?,?,?,?,?,?,?)",
                (call_id, other["object_id"], w.principal.user_id, "lease-other",
                 "completed", 37, response_hash, w.clock.ms),
            )
    elif kind != "missing":
        w.journal(status="failed" if kind == "failed" else "completed", tokens=37,
                  action=action, call_id=call_id)
    changes = {"pending_action": {"call_id": call_id, "action": action},
               "status": "running", "lease_id": "lease-r5-crash",
               "lease_until_ms": w.clock.ms - 1}
    if kind == "already_handled":
        changes["handled_call_ids"] = [call_id]
    w.patch(**changes)
    task = w.run()
    assert task["status"] == "blocked"
    assert task["payload"]["tools"] == 0
    assert w.provider.contexts == []
    assert_no_recommendation(w)


def test_all_budget_limits_are_the_documented_fixed_bounds():
    assert (module.MAX_CALLS, module.MAX_TOOLS, module.MAX_TOKENS) == (14, 12, 32000)
    assert (module.MAX_SECONDS, module.LEASE_SECONDS) == (420.0, 900.0)


@pytest.mark.parametrize("seconds", [float("nan"), float("inf")])
def test_invalid_monotonic_clock_before_dispatch_does_not_create_call(world, seconds):
    w = world()
    w.clock.seconds = seconds
    assert_error("clock_invalid", w.run)
    assert w.rows() == [] and w.provider.contexts == []
    assert_no_recommendation(w)


def test_predispatch_clock_failure_is_known_zero_usage_not_unknown(world):
    w = world()
    values = iter([100.0, 100.0, float("nan"), 100.0, 100.0])
    w.engine.monotonic = lambda: next(values, 100.0)
    task = w.run()
    assert task["status"] == "blocked"
    assert task["payload"]["error"] == "clock_invalid"
    assert w.provider.contexts == []
    assert len(w.rows()) == 1
    assert w.rows()[0]["status"] == "failed"
    assert w.rows()[0]["tokens"] == 0
    assert task["payload"]["unknown_calls"] == 0
    assert_no_recommendation(w)


@pytest.mark.parametrize("tokens", [None, True, -1, 100001, float("nan"), "128"])
def test_untrusted_usage_never_becomes_a_free_success(world, tokens):
    def corrupt_usage(_context, reply):
        # An untrusted provider implementation bypasses its own reply validator.
        object.__setattr__(reply, "tokens", tokens)
        return reply

    w = world(provider=ChoiceProvider(corrupt_usage))
    task = w.run()
    assert task["status"] == "blocked"
    assert task["payload"]["tools"] == 0
    assert task["payload"]["unknown_calls"] == 1
    assert len(w.rows()) == 1
    assert w.rows()[0]["status"] == "unknown"
    assert w.rows()[0]["tokens"] is None
    assert w.rows()[0]["response_hash"] is None
    assert_no_recommendation(w)


def test_supplement_retains_exhausted_call_budget_and_does_not_autorun(world):
    w = world("missing")
    waiting = w.run()
    assert waiting["status"] == "awaiting_input"
    for index in range(module.MAX_CALLS - waiting["payload"]["calls"]):
        w.journal(status="failed", tokens=0, call_id=f"call-prior-{index}")
    w.patch(elapsed_s=400.0)
    resumed = w.engine.resume(w.principal, w.task_id, action_id="r5-supplement-budget",
                              fields={"material_viscosity": 1.05, "ambient_c": 23.0},
                              note="Missing values explicitly supplied; old usage remains.")
    assert resumed["payload"]["calls"] == module.MAX_CALLS
    assert resumed["payload"]["elapsed_s"] == 400.0
    assert resumed["payload"]["tokens"] == waiting["payload"]["tokens"]
    assert len(w.provider.contexts) == 2
    try:
        task = w.run()
    except DomainError as exc:
        assert exc.code == "budget_exhausted"
    else:
        assert task["status"] == "blocked" and task["payload"]["error"] == "budget_exhausted"
    assert len(w.provider.contexts) == 2
    assert len(w.rows()) == module.MAX_CALLS
    assert_no_recommendation(w)


@pytest.mark.parametrize("known_tokens", [0, 79])
def test_known_usage_unknown_outcome_requires_reconciliation(world, known_tokens):
    def unknown_outcome(_context, _reply):
        raise ProviderError("provider_unknown", "TEST outcome unknown despite reliable usage",
                            tokens=known_tokens, sent=True)

    w = world(provider=ChoiceProvider(unknown_outcome))
    task = w.run()
    assert task["status"] == "blocked"
    assert task["payload"]["tokens"] == known_tokens
    assert task["payload"]["unknown_calls"] == 1
    assert task["payload"]["tools"] == 0
    assert task["payload"]["pending_action"] is None
    row = w.rows()[0]
    assert row["status"] == "unknown"
    assert row["tokens"] == known_tokens
    assert row["error_code"] == "provider_unknown"
    assert row["response_hash"] is None
    w.restart()
    assert w.run()["status"] == "blocked"
    assert len(w.provider.contexts) == len(w.rows()) == 1
    if known_tokens:
        assert_error("invalid_usage", w.engine.reconcile, w.principal, w.task_id,
                     action_id="r5-understated-unknown", call_id=row["call_id"], no_result=True,
                     tokens=known_tokens - 1, note="TEST usage cannot reduce the known charge.")
        assert w.rows()[0]["status"] == "unknown"
    settled = w.engine.reconcile(
        w.principal, w.task_id, action_id="r5-settle-known-unknown", call_id=row["call_id"],
        no_result=True, tokens=known_tokens, note="TEST no result recovered; usage was already known.",
    )
    assert settled["status"] == "created"
    assert settled["payload"]["unknown_calls"] == 0
    assert settled["payload"]["tokens"] == known_tokens
    assert settled["payload"]["evidence"] == []
    assert settled["payload"]["pending_action"] is None
    assert_no_recommendation(w)
    w.provider.hook = None
    completed = w.run()
    assert completed["status"] == "submitted"
    assert completed["payload"]["calls"] == 7
    assert completed["payload"]["tokens"] == known_tokens + 6 * 128


@pytest.mark.parametrize("checkpoint", [30.0, 415.0])
def test_crash_accounting_adds_only_uncheckpointed_call_elapsed(world, checkpoint):
    w = world()
    w.journal(status="failed", tokens=11, elapsed=4.0, call_id="call-r5-checkpointed")
    # The checkpoint includes four seconds of an old call plus non-call work.
    # Only the new call below commits after this last task checkpoint.
    w.patch(elapsed_s=checkpoint, accounted_call_elapsed_s=4.0, status="running",
            lease_id="lease-r5-crash", lease_until_ms=w.clock.ms - 1)
    action = {"tool": "get_process_context", "arguments": {}}
    w.journal(status="completed", tokens=37, elapsed=9.0, action=action)
    w.restart()
    expected = checkpoint + 9.0
    for _ in range(3):
        assert w.get()["payload"]["elapsed_s"] == expected
    assert w.provider.contexts == []
    assert sum(row["elapsed_s"] for row in w.rows()) == 13.0
    if expected >= module.MAX_SECONDS:
        assert_error("budget_exhausted", w.run)
        assert w.provider.contexts == []
        assert len(w.rows()) == 2
        assert_no_recommendation(w)
    else:
        task = w.run()
        assert task["status"] == "submitted"
        assert task["payload"]["elapsed_s"] == expected
        assert task["payload"]["calls"] == 7
        assert task["payload"]["tools"] == 6
        assert task["payload"]["tokens"] == 11 + 37 + 5 * 128
        assert len(w.provider.contexts) == 5
        # Saving the recovered watermark and reopening must not charge it twice.
        w.restart()
        for _ in range(3):
            assert w.get()["payload"]["elapsed_s"] == expected
        assert w.run()["payload"]["elapsed_s"] == expected
        assert len(w.rows()) == 7


def test_live_elapsed_checkpoint_does_not_double_count_already_included_call_time(world):
    w = None

    def advance(_context, reply):
        w.clock.seconds += 3.0
        return reply

    w = world(provider=ChoiceProvider(advance))
    # Earlier graph bookkeeping took ten seconds; all new calls consume three.
    w.patch(elapsed_s=10.0, accounted_call_elapsed_s=0.0)
    task = w.run()
    assert task["status"] == "submitted"
    assert sum(row["elapsed_s"] for row in w.rows()) == 18.0
    assert task["payload"]["elapsed_s"] == 28.0
    w.restart()
    for _ in range(3):
        assert w.get()["payload"]["elapsed_s"] == 28.0
    assert w.run()["payload"]["elapsed_s"] == 28.0
    assert len(w.provider.contexts) == 6


@pytest.mark.parametrize("normalize_via_run", [False, True])
def test_cancelled_crashed_call_can_be_normalized_and_reconciled(world, normalize_via_run):
    w = world()
    call_id = w.journal(status="started")
    w.patch(status="running", lease_id="lease-r5-crash",
            lease_until_ms=w.clock.ms + int(module.LEASE_SECONDS * 1000))
    assert w.engine.cancel(w.principal, w.task_id, action_id="r5-cancel-crashed")["status"] == "cancelled"
    kwargs = dict(action_id="r5-reconcile-crashed-cancel", call_id=call_id, no_result=True,
                  tokens=41, note="TEST worker died during a cancelled call; no result recovered.")
    assert_error("busy", w.engine.reconcile, w.principal, w.task_id, **kwargs)
    assert w.rows()[0]["status"] == "started"
    w.clock.ms += int(module.LEASE_SECONDS * 1000) + 1
    w.restart()
    if normalize_via_run:
        task = w.run()
        assert task["status"] == "cancelled"
        assert task["payload"]["unknown_calls"] == 1
        assert task["payload"]["lease_id"] is None
        assert w.rows()[0]["status"] == "unknown"
        assert w.rows()[0]["tokens"] is None
    settled = w.engine.reconcile(w.principal, w.task_id, **kwargs)
    assert settled["status"] == "cancelled"
    assert settled["payload"]["unknown_calls"] == 0
    assert settled["payload"]["tokens"] == 41
    assert settled["payload"]["lease_id"] is None
    assert settled["payload"]["pending_action"] is None
    assert settled["payload"]["evidence"] == []
    row = w.rows()[0]
    assert row["status"] == "failed"
    assert row["error_code"] == "reconciled_no_result"
    assert row["tokens"] == 41
    assert row["response_hash"] is None
    assert w.engine.reconcile(w.principal, w.task_id, **kwargs) == settled
    assert w.run()["status"] == "cancelled"
    assert w.provider.contexts == []
    assert len(w.rows()) == 1
    assert_no_recommendation(w)


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("settled_tokens,late_tokens,settled_elapsed,late_elapsed", [
    (29, 83, 8.0, 12.0), (83, 29, 12.0, 8.0),
])
def test_late_reply_after_reconciliation_only_increases_accounting(
    world, cancelled, settled_tokens, late_tokens, settled_elapsed, late_elapsed,
):
    entered, release = threading.Event(), threading.Event()

    def pause(_context, reply):
        entered.set()
        assert release.wait(10), "Late provider reply was not released"
        return ProviderReply(reply.action, late_tokens)

    w = world(provider=ChoiceProvider(pause))
    recovery = AgentEngine(w.service, w.provider, clock=w.clock.wall,
                           monotonic=w.clock.monotonic)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(w.run)
        try:
            assert entered.wait(10)
            call_id = w.rows()[0]["call_id"]
            w.clock.ms += int(module.LEASE_SECONDS * 1000) + 1
            assert recovery.run(w.principal, w.task_id)["status"] == "blocked"
            assert w.rows()[0]["status"] == "unknown"
            if cancelled:
                recovery.cancel(w.principal, w.task_id, action_id="r5-cancel-before-settle")
            # Durable elapsed measurement already present before human settlement.
            with w.store.transaction() as conn:
                conn.execute("UPDATE calls SET elapsed_s=? WHERE call_id=?",
                             (settled_elapsed, call_id))
            reconcile_kwargs = dict(
                action_id="r5-settle-before-late", call_id=call_id,
                no_result=True, tokens=settled_tokens,
                note="TEST expired worker had no recovered response at settlement time.",
            )
            settled = recovery.reconcile(w.principal, w.task_id, **reconcile_kwargs)
            assert settled["status"] == ("cancelled" if cancelled else "created")
            assert w.rows()[0]["status"] == "failed"
            assert w.rows()[0]["tokens"] == settled_tokens
            w.clock.seconds += late_elapsed
        finally:
            release.set()
        result = future.result(timeout=10)
    row = w.rows()[0]
    assert row["status"] == "failed"
    assert row["error_code"] == "reconciled_no_result"
    assert row["tokens"] == max(settled_tokens, late_tokens)
    assert row["elapsed_s"] == max(settled_elapsed, late_elapsed)
    assert row["response_hash"] is None
    assert result["status"] == ("cancelled" if cancelled else "created")
    assert result["payload"]["tokens"] == max(settled_tokens, late_tokens)
    assert result["payload"]["unknown_calls"] == 0
    assert result["payload"]["tools"] == 0
    assert result["payload"]["pending_action"] is None
    assert result["payload"]["evidence"] == []
    assert len(w.provider.contexts) == len(w.rows()) == 1
    w.restart()
    for _ in range(3):
        task = w.get()
        assert task["payload"]["tokens"] == max(settled_tokens, late_tokens)
        assert task["payload"]["elapsed_s"] == max(settled_elapsed, late_elapsed)
    # A receipt replays its original response, not the later accounting view.
    # Replaying must never reapply the old lower charge or mutate settled state.
    assert w.engine.reconcile(w.principal, w.task_id, **reconcile_kwargs) == settled
    assert settled["payload"]["tokens"] == settled_tokens
    assert settled["payload"]["elapsed_s"] == settled_elapsed
    assert_error("idempotency_conflict", w.engine.reconcile, w.principal, w.task_id,
                 **{**reconcile_kwargs, "tokens": settled_tokens + 1})
    assert w.rows()[0] == row
    assert w.get()["payload"]["tokens"] == max(settled_tokens, late_tokens)
    assert w.get()["payload"]["elapsed_s"] == max(settled_elapsed, late_elapsed)
    assert_no_recommendation(w)


@pytest.mark.parametrize("identity", ["task_id", "owner", "lease_id"])
def test_late_reconciled_accounting_cannot_bypass_journal_identity_fence(world, identity):
    w = None
    settled_row = None

    def settle_and_tamper(_context, reply):
        nonlocal settled_row
        row = w.rows()[0]
        with w.store.transaction() as conn:
            conn.execute(
                "UPDATE calls SET status='unknown',tokens=17,elapsed_s=2.0,"
                "error_code='interrupted_call' WHERE call_id=?", (row["call_id"],),
            )
        w.patch(status="blocked", lease_id=None, lease_until_ms=0)
        w.engine.reconcile(w.principal, w.task_id, action_id="r5-identity-settlement",
                           call_id=row["call_id"], no_result=True, tokens=17,
                           note="TEST no response recovered before journal identity tampering.")
        # Only test-controlled column names enter this crash/corruption fixture.
        with w.store.transaction() as conn:
            conn.execute(f"UPDATE calls SET {identity}=? WHERE call_id=?",
                         ("r5-replaced-identity", row["call_id"]))
            settled_row = dict(conn.execute("SELECT * FROM calls WHERE call_id=?",
                                            (row["call_id"],)).fetchone())
        return ProviderReply(reply.action, 97)

    w = world(provider=ChoiceProvider(settle_and_tamper))
    task = w.run()
    with w.store.connect() as conn:
        after = dict(conn.execute("SELECT * FROM calls WHERE call_id=?",
                                  (settled_row["call_id"],)).fetchone())
    assert after == settled_row
    assert after["tokens"] == 17 and after["elapsed_s"] == 2.0
    assert after["status"] == "failed" and after["response_hash"] is None
    assert task["status"] == "created"
    assert task["payload"]["tools"] == 0
    assert task["payload"]["pending_action"] is None
    assert task["payload"]["evidence"] == []
    assert len(w.provider.contexts) == 1
    assert_no_recommendation(w)


def test_time_budget_expires_at_last_predispatch_check_without_sending(world):
    w = world()
    checks_after_journal = 0

    def monotonic():
        nonlocal checks_after_journal
        if w.rows():
            checks_after_journal += 1
            # First post-journal read is call_start; the next is _elapsed just
            # before decide. That final predispatch check reaches the exact cap.
            if checks_after_journal >= 2:
                w.clock.seconds = 100.0 + module.MAX_SECONDS
        return w.clock.seconds

    w.engine.monotonic = monotonic
    task = w.run()
    assert task["status"] == "blocked"
    assert task["payload"]["error"] == "budget_exhausted"
    assert task["payload"]["elapsed_s"] == module.MAX_SECONDS
    assert task["payload"]["calls"] == 1
    assert task["payload"]["tokens"] == 0
    assert task["payload"]["unknown_calls"] == 0
    assert task["payload"]["tools"] == 0
    assert task["payload"]["pending_action"] is None
    assert w.provider.contexts == []
    row = w.rows()[0]
    assert row["status"] == "failed"
    assert row["tokens"] == 0
    assert row["response_hash"] is None
    assert_no_recommendation(w)


def test_old_lease_elapsed_correction_adds_to_new_run_and_real_tool_time(world, monkeypatch):
    w = None
    old_engine = None
    old_call_id = None
    corrected = False
    retrieved = []

    def late_elapsed(_context, reply):
        nonlocal corrected
        if not corrected:
            corrected = True
            assert reply.action["tool"] == "get_process_context"
            # This newer provider call really consumes ten injected seconds.
            w.clock.seconds += 10.0
            old_engine._account(
                old_call_id, w.task_id, w.principal.user_id, "lease-r5-crash",
                status="completed", tokens=11, elapsed=30.0,
                response={"action": {"tool": "get_process_context", "arguments": {}}},
            )
        return reply

    w = world(provider=ChoiceProvider(late_elapsed))
    old_call_id = w.journal(status="unknown", tokens=11)
    w.patch(status="blocked")
    w.engine.reconcile(w.principal, w.task_id, action_id="r5-settle-before-new-run",
                       call_id=old_call_id, no_result=True, tokens=11,
                       note="TEST old worker had no response when its outcome was settled.")
    w.patch(elapsed_s=350.0, accounted_call_elapsed_s=0.0)
    old_engine = w.engine
    w.restart()
    actual_experience = w.service.experience

    def timed_actual_experience(principal, context_id):
        # Latency instrumentation only: execute the real owner-scoped retrieval,
        # retain its actual result, then account forty seconds of tool work.
        result = actual_experience(principal, context_id)
        retrieved.append(copy.deepcopy(result))
        w.clock.seconds += 40.0
        return result

    monkeypatch.setattr(w.service, "experience", timed_actual_experience)
    task = w.run()
    assert corrected is True
    assert len(retrieved) == 1
    assert retrieved[0] == actual_experience(w.principal, w.context.context_id)
    assert retrieved[0]["items"]
    assert retrieved[0]["context_hash"] == digest(w.context.model_dump(mode="json"))
    assert task["status"] == "blocked"
    assert task["payload"]["error"] == "budget_exhausted"
    assert task["payload"]["elapsed_s"] == 430.0
    assert task["payload"]["calls"] == 3
    assert task["payload"]["tokens"] == 11 + 2 * 128
    assert task["payload"]["tools"] == 1
    assert {item["tool"] for item in task["payload"]["evidence"]} == {"get_process_context"}
    rows = w.rows()
    assert rows[0]["call_id"] == old_call_id
    assert rows[0]["status"] == "failed"
    assert rows[0]["error_code"] == "reconciled_no_result"
    assert rows[0]["response_hash"] is None
    assert [row["elapsed_s"] for row in rows] == [30.0, 10.0, 0.0]
    assert len(w.provider.contexts) == 2
    assert_no_recommendation(w)
    w.restart()
    for _ in range(3):
        assert w.get()["payload"]["elapsed_s"] == 430.0
    assert_error("budget_exhausted", w.run)
    assert len(w.provider.contexts) == 2


def test_late_old_usage_at_reservation_blocks_new_dispatch_at_token_cap(world):
    w = world()
    old_call_id = w.journal(status="unknown", tokens=11)
    w.patch(status="blocked")
    reconcile_kwargs = dict(
        action_id="r5-settle-before-reservation", call_id=old_call_id,
        no_result=True, tokens=11,
        note="TEST no old response recovered before a newer run reserved a call.",
    )
    settled = w.engine.reconcile(w.principal, w.task_id, **reconcile_kwargs)
    old_engine = w.engine
    w.restart()
    injected = False
    reserved_call_id = None

    def monotonic():
        nonlocal injected, reserved_call_id
        started = [row for row in w.rows() if row["status"] == "started"]
        if started and not injected:
            injected = True
            assert len(started) == 1
            reserved_call_id = started[0]["call_id"]
            assert reserved_call_id != old_call_id
            assert w.provider.contexts == []
            # The old worker reports its reliable charge after the new journal
            # reservation, at call_start, before the predispatch reload.
            old_engine._account(
                old_call_id, w.task_id, w.principal.user_id, "lease-r5-crash",
                status="completed", tokens=module.MAX_TOKENS, elapsed=0.0,
                response={"action": {"tool": "get_process_context", "arguments": {}}},
            )
        return w.clock.seconds

    w.engine.monotonic = monotonic
    task = w.run()
    assert injected is True
    assert task["status"] == "blocked"
    assert task["payload"]["error"] == "budget_exhausted"
    assert task["payload"]["tokens"] == module.MAX_TOKENS
    assert task["payload"]["unknown_calls"] == 0
    assert task["payload"]["tools"] == 0
    assert task["payload"]["evidence"] == []
    assert task["payload"]["pending_action"] is None
    assert w.provider.contexts == []
    rows = w.rows()
    assert len(rows) == 2
    assert rows[0]["call_id"] == old_call_id
    assert rows[0]["status"] == "failed"
    assert rows[0]["error_code"] == "reconciled_no_result"
    assert rows[0]["tokens"] == module.MAX_TOKENS
    assert rows[0]["response_hash"] is None
    assert rows[1]["call_id"] == reserved_call_id
    assert rows[1]["status"] == "failed"
    assert rows[1]["error_code"] == "budget_exhausted"
    assert rows[1]["tokens"] == 0
    assert rows[1]["response_hash"] is None
    assert w.engine.reconcile(w.principal, w.task_id, **reconcile_kwargs) == settled
    assert w.rows() == rows
    w.restart()
    for _ in range(3):
        assert w.get()["payload"]["tokens"] == module.MAX_TOKENS
    assert_error("budget_exhausted", w.run)
    assert w.provider.contexts == []
    assert len(w.rows()) == 2
    assert_no_recommendation(w)


def test_stop_elapsed_sampling_and_save_atomically_include_late_old_lease_time(world, monkeypatch):
    w = None

    def billed_failure(_context, _reply):
        w.clock.seconds += 10.0
        raise ProviderError("provider_invalid", "TEST billed failure before slow session cleanup",
                            tokens=37)

    class SlowCleanupProvider(SessionProvider):
        @contextmanager
        def session(self):
            try:
                with super().session():
                    yield self
            finally:
                # Non-call cleanup work has no call-journal duration and has not
                # reached a task checkpoint when the outer stop handler runs.
                w.clock.seconds += 40.0

    provider = SlowCleanupProvider(billed_failure)
    w = world(provider=provider)
    old_call_id = w.journal(status="unknown", tokens=11)
    w.patch(status="blocked")
    w.engine.reconcile(
        w.principal, w.task_id, action_id="r5-settle-before-atomic-stop",
        call_id=old_call_id, no_result=True, tokens=11,
        note="TEST old worker had no result before the new run started.",
    )
    w.patch(elapsed_s=350.0, accounted_call_elapsed_s=0.0)
    old_engine = w.engine
    w.restart()
    actual_stop = w.engine._stop
    injections = []
    samples = []

    def inject_before_stop_transaction(principal, task_id, lease_id, code, elapsed=None):
        # The former implementation supplied a pre-sampled 400-second scalar
        # here. Injecting the old worker's +30 before stop loads its checkpoint
        # exposed max(390, 400), losing thirty seconds of disjoint elapsed work.
        # Keep the real _stop implementation; only schedule this late journal
        # update deterministically at its transaction boundary.
        assert code == "provider_invalid"
        assert w.clock.seconds == 150.0
        old_engine._account(
            old_call_id, task_id, principal.user_id, "lease-r5-crash",
            status="completed", tokens=11, elapsed=30.0,
            response={"action": {"tool": "get_process_context", "arguments": {}}},
        )
        injections.append(old_call_id)
        if callable(elapsed):
            actual_sample = elapsed

            def sample_inside_transaction(conn):
                value = actual_sample(conn)
                old_duration = conn.execute("SELECT elapsed_s FROM calls WHERE call_id=?",
                                            (old_call_id,)).fetchone()[0]
                samples.append((conn.in_transaction, old_duration, value))
                return value

            elapsed = sample_inside_transaction
        return actual_stop(principal, task_id, lease_id, code, elapsed)

    monkeypatch.setattr(w.engine, "_stop", inject_before_stop_transaction)
    task = w.run()
    assert injections == [old_call_id]
    assert task["status"] == "blocked"
    assert task["payload"]["error"] == "provider_invalid"
    assert task["payload"]["elapsed_s"] == 430.0
    assert samples == [(True, 30.0, 430.0)]
    assert task["payload"]["accounted_call_elapsed_s"] == 40.0
    assert task["payload"]["tokens"] == 48
    assert task["payload"]["unknown_calls"] == 0
    assert task["payload"]["tools"] == 0
    assert task["payload"]["pending_action"] is None
    assert task["payload"]["evidence"] == []
    rows = w.rows()
    assert len(rows) == 2
    assert rows[0]["call_id"] == old_call_id
    assert rows[0]["status"] == "failed"
    assert rows[0]["error_code"] == "reconciled_no_result"
    assert rows[0]["tokens"] == 11
    assert rows[0]["elapsed_s"] == 30.0
    assert rows[0]["response_hash"] is None
    assert rows[1]["status"] == "failed"
    assert rows[1]["error_code"] == "provider_invalid"
    assert rows[1]["tokens"] == 37
    assert rows[1]["elapsed_s"] == 10.0
    assert rows[1]["response_hash"] is None
    assert provider.sessions == provider.exits == 1
    assert len(provider.contexts) == 1
    assert_no_recommendation(w)
    w.restart()
    for _ in range(3):
        assert w.get()["payload"]["elapsed_s"] == 430.0
    assert_error("budget_exhausted", w.run)
    assert w.rows() == rows
    assert len(provider.contexts) == 1


def test_create_cancel_replay_is_owner_scoped_and_binds_target(world):
    w = world()
    original = w.get()
    assert w.engine.create(w.principal, w.context.context_id, action_id="r5-create") == original
    cancelled = w.engine.cancel(w.principal, w.task_id, action_id="r5-cancel")
    assert w.engine.cancel(w.principal, w.task_id, action_id="r5-cancel") == cancelled
    assert w.run()["status"] == "cancelled"
    assert w.provider.contexts == []
    other = w.store.create_user("r5-other", "r5-other-test-token-" + "y" * 32,
                                "admin", [w.context.plant_id])
    with pytest.raises(DomainError) as caught:
        w.engine.get(other, w.task_id)
    assert caught.value.status == 404
    w.store.set_user(w.principal.user_id, active=False)
    with pytest.raises(DomainError):
        w.engine.cancel(w.principal, w.task_id, action_id="r5-cancel")
