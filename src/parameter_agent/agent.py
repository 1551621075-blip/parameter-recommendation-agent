"""Bounded single-agent graph. Only tools may create authoritative evidence."""

import copy
import math
import threading
import time
import uuid
from contextlib import contextmanager
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context

from .bc import predict
from .domain import ProcessContext, check_candidate, digest, validate_context
from .errors import DomainError
from .provider import ProviderError, ProviderReply

MAX_CALLS = 14
MAX_TOOLS = 12
MAX_TOKENS = 32000
MAX_SECONDS = 420.0
LEASE_SECONDS = 900.0
TERMINAL = {"submitted", "no_recommendation", "cancelled"}
SOURCE_KEYS = (
    "context_hash", "context_version", "model_hash", "model_content_hash", "model_version",
    "experience_version", "constraint_version",
)
TOOLS = (
    "get_process_context", "retrieve_experience", "predict_bc", "draft_recommendation",
    "check_constraints", "request_human_input", "submit_recommendation",
)


class GraphState(TypedDict):
    task_id: str
    stop: bool


def _identifier(prefix):
    return prefix + uuid.uuid4().hex


def _note(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise DomainError("invalid_note", "A bounded nonempty note is required", 422)
    return value


class AgentEngine:
    def __init__(self, service, provider, *, clock=None, monotonic=None):
        self.service = service
        self.store = service.store
        self.provider = provider
        self.clock = clock or (lambda: int(time.time() * 1000))
        self.monotonic = monotonic or time.monotonic
        self._provider_lock = threading.Lock()

    def _now(self):
        value = self.clock()
        if type(value) is not int or not 0 <= value <= 9007199254740991:
            raise DomainError("clock_invalid", "Invalid local clock")
        return value

    def _usage(self, conn, task_id, owner):
        rows = conn.execute(
            "SELECT status,tokens,elapsed_s FROM calls WHERE task_id=? AND owner=?",
            (task_id, owner),
        ).fetchall()
        return {
            "calls": len(rows),
            "tokens": sum(row["tokens"] or 0 for row in rows),
            "unknown_calls": sum(row["status"] in {"started", "unknown"} for row in rows),
            "call_elapsed_s": sum(row["elapsed_s"] for row in rows),
        }

    def _load(self, conn, principal, task_id):
        obj = self.store.get_object("task", task_id, principal, conn=conn)
        p = obj["payload"]
        usage = self._usage(conn, task_id, principal.user_id)
        p["calls"] = usage["calls"]
        p["tokens"] = usage["tokens"]
        p["unknown_calls"] = usage["unknown_calls"]
        accounted = p.get("accounted_call_elapsed_s", 0.0)
        p["elapsed_s"] += max(0.0, usage["call_elapsed_s"] - accounted)
        p["accounted_call_elapsed_s"] = usage["call_elapsed_s"]
        return obj

    def _save(self, conn, principal, obj, p):
        return self.store.update_object(
            conn, kind="task", object_id=obj["object_id"], principal=principal,
            expected_version=obj["version"], status=p["status"], payload=p,
        )

    def _audit(self, conn, principal, obj, action, detail):
        self.store.audit(conn, principal, kind="task", object_id=obj["object_id"],
                         action=action, detail=detail)

    def _sources(self, snapshot):
        return {key: snapshot[key] for key in SOURCE_KEYS}

    def _snapshot(self, conn, principal, p):
        snapshot = self.service.snapshot(principal, p["context_id"], conn=conn)
        if self._sources(snapshot) != p["source_snapshot"]:
            raise DomainError("source_changed", "Context or model source changed; create a new task")
        return snapshot

    def create(self, principal, context_id, *, action_id):
        request = {"operation": "create_task", "context_id": context_id}
        with self.store.transaction() as conn:
            context = self.service.get_context(principal, context_id, conn=conn)
            old = self.store.receipt(conn, principal, action_id, request)
            if old is not None:
                return old
            snapshot = self.service.snapshot(principal, context_id, conn=conn)
            task_id = _identifier("task-")
            payload = {
                "context_id": context_id, "source_snapshot": self._sources(snapshot),
                "evidence": [], "prediction": None, "candidate": None, "draft": None,
                "status": "created", "calls": 0, "tools": 0, "tokens": 0,
                "unknown_calls": 0, "elapsed_s": 0.0, "accounted_call_elapsed_s": 0.0,
                "error": None,
                "pending_action": None, "awaiting_fields": [], "lease_id": None,
                "lease_until_ms": 0, "provider_fingerprint": self.provider.fingerprint,
                "synthetic": True, "recommendation_id": None, "handled_call_ids": [],
            }
            obj = self.store.create_object(
                conn, kind="task", object_id=task_id, principal=principal,
                plant_id=context["plant_id"], status="created", payload=payload,
            )
            self._audit(conn, principal, obj, "task_created", {"context_id": context_id})
            self.store.record_receipt(conn, principal, action_id, request, obj)
            return obj

    def get(self, principal, task_id):
        with self.store.connect() as conn:
            return self._load(conn, principal, task_id)

    def _fence(self, obj, lease_id):
        p = obj["payload"]
        if p["lease_id"] != lease_id or p["status"] != "running":
            raise DomainError("worker_stopped", "Task was stopped or replaced")
        if self._now() >= p["lease_until_ms"]:
            raise DomainError("lease_expired", "Worker lease expired")

    def _budget(self, p, *, call=False, tool=False):
        if p["unknown_calls"]:
            raise DomainError("model_unknown", "Unknown model result requires reconciliation")
        if p["tokens"] >= MAX_TOKENS or p["elapsed_s"] >= MAX_SECONDS:
            raise DomainError("budget_exhausted", "Task budget exhausted")
        if (call and p["calls"] >= MAX_CALLS) or ((call or tool) and p["tools"] >= MAX_TOOLS):
            raise DomainError("budget_exhausted", "Task budget exhausted")

    def _available(self, p, snapshot):
        issued = {e["tool"] for e in p["evidence"]}
        if "get_process_context" not in issued:
            return ["get_process_context"]
        ctx = ProcessContext.model_validate(snapshot["context"])
        missing = [key for key in ("material_viscosity", "ambient_c") if getattr(ctx, key) is None]
        if missing:
            return ["request_human_input"]
        if p["prediction"] is not None and p["prediction"]["status"] != "ok":
            return []
        menu = []
        if "retrieve_experience" not in issued:
            menu.append("retrieve_experience")
        if "predict_bc" not in issued:
            menu.append("predict_bc")
        if {"retrieve_experience", "predict_bc"} <= issued:
            if p["draft"] is None:
                menu.append("draft_recommendation")
            elif p["candidate"] is None:
                menu.append("check_constraints")
            elif p["candidate"]["eligible"]:
                menu.append("submit_recommendation")
        return menu

    def _model_context(self, task_id, p, snapshot):
        ctx = snapshot["context"]
        return {
            "task_id": task_id, "context_hash": p["source_snapshot"]["context_hash"],
            "context": ctx, "model_hash": snapshot["model_hash"],
            "available_tools": self._available(p, snapshot),
            "evidence": {e["tool"]: e for e in p["evidence"]},
            "evidence_ids": [e["evidence_id"] for e in p["evidence"]],
            "candidate": p["candidate"], "prediction": p["prediction"],
            "required_input": [k for k in ("material_viscosity", "ambient_c") if ctx[k] is None],
            "boundary": "Synthetic reference only. BC imitates expert actions, not effects. "
                        "Submission creates pending review, never approval or device execution.",
        }

    def _pending_action(self, conn, principal, task_id, p):
        pending = p["pending_action"]
        if not isinstance(pending, dict) or set(pending) != {"call_id", "action"}:
            raise DomainError("integrity", "Invalid pending model action")
        if pending["call_id"] in p["handled_call_ids"]:
            raise DomainError("integrity", "Model response has already been consumed")
        row = conn.execute(
            "SELECT status,response_hash FROM calls WHERE call_id=? AND task_id=? AND owner=?",
            (pending["call_id"], task_id, principal.user_id),
        ).fetchone()
        if row is None or row["status"] != "completed" or not row["response_hash"]:
            raise DomainError("integrity", "Pending action is not bound to a completed response")
        response = self.store.get_artifact(row["response_hash"])
        if set(response) != {"action"} or response["action"] != pending["action"]:
            raise DomainError("integrity", "Pending action differs from recorded model response")
        return pending["action"]

    def _validate_action(self, action, p, snapshot):
        if not isinstance(action, dict) or set(action) != {"tool", "arguments"}:
            raise DomainError("invalid_action", "Model action has an invalid shape")
        tool, args = action["tool"], action["arguments"]
        if not isinstance(tool, str) or tool not in self._available(p, snapshot):
            raise DomainError("tool_unavailable", "Tool is not available in this state")
        if not isinstance(args, dict):
            raise DomainError("invalid_action", "Tool arguments must be an object")
        if tool == "request_human_input":
            if set(args) != {"question", "fields"}:
                raise DomainError("invalid_action", "Invalid human input request")
            _note(args["question"])
            fields = args["fields"]
            required = self._model_context("", p, snapshot)["required_input"]
            if not isinstance(fields, list) or sorted(fields) != sorted(required):
                raise DomainError("invalid_action", "Human input fields do not match missing context")
        elif tool == "submit_recommendation":
            if set(args) != {"candidate_hash", "evidence_ids", "explanation"}:
                raise DomainError("invalid_action", "Invalid submission arguments")
            _note(args["explanation"])
            ids = args["evidence_ids"]
            expected = [e["evidence_id"] for e in p["evidence"]]
            if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
                raise DomainError("invalid_evidence", "Invalid evidence references")
            if sorted(ids) != sorted(expected):
                raise DomainError("invalid_evidence", "Submission must cite all issued evidence exactly")
            if args["candidate_hash"] != p["candidate"]["candidate_hash"]:
                raise DomainError("candidate_changed", "Submission candidate does not match frozen hash")
        elif args:
            raise DomainError("invalid_action", "This tool accepts no arguments")
        return tool, args

    def _sign(self, task_id, p, tool, payload):
        evidence = {
            "evidence_id": _identifier("ev-"), "task_id": task_id, "tool": tool,
            "context_hash": p["source_snapshot"]["context_hash"], "payload": payload,
        }
        evidence["content_hash"] = digest(evidence)
        return evidence

    def _elapsed(self, started, base):
        value = self.monotonic()
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < started:
            raise DomainError("clock_invalid", "Invalid monotonic clock")
        return base + value - started

    def _account(self, call_id, task_id, owner, lease_id, *, status, tokens, elapsed, response=None,
                 error=None):
        if tokens is not None and (type(tokens) is not int or not 0 <= tokens <= 100000):
            status, tokens, response, error = "unknown", None, None, "invalid_usage"
        response_hash = self.store.put_artifact(response) if response is not None else None
        with self.store.transaction() as conn:
            row = conn.execute(
                "SELECT task_id,owner,lease_id,status,tokens,elapsed_s,error_code FROM calls "
                "WHERE call_id=?", (call_id,),
            ).fetchone()
            if row is None or (row["task_id"], row["owner"], row["lease_id"]) != (
                task_id, owner, lease_id,
            ):
                raise DomainError("call_fence", "Model journal ownership changed")
            if row["status"] == "failed" and row["error_code"] == "reconciled_no_result":
                # A late original worker may increase known cost, never resurrect its action.
                conn.execute(
                    "UPDATE calls SET tokens=?,elapsed_s=? WHERE call_id=?",
                    (max(row["tokens"] or 0, tokens or 0), max(row["elapsed_s"], elapsed), call_id),
                )
                return
            if row["status"] not in {"started", "unknown"}:
                raise DomainError("call_fence", "Model journal was already settled")
            cursor = conn.execute(
                "UPDATE calls SET status=?,tokens=?,elapsed_s=?,response_hash=?,error_code=? "
                "WHERE call_id=? AND task_id=? AND owner=? AND lease_id=? "
                "AND (status='started' OR (status='unknown' AND error_code='interrupted_call' "
                "AND tokens IS NULL))",
                (status, tokens, elapsed, response_hash, error, call_id, task_id, owner, lease_id),
            )
            if cursor.rowcount != 1:
                raise DomainError("call_fence", "Model journal ownership changed")

    def _stop(self, principal, task_id, lease_id, code, elapsed=None):
        with self.store.transaction() as conn:
            if callable(elapsed):
                try:
                    elapsed = elapsed(conn)
                except Exception:
                    elapsed = None
            obj = self._load(conn, principal, task_id)
            p = obj["payload"]
            if p["lease_id"] != lease_id:
                if code in {"provider_unknown", "provider_cleanup_unknown"} and (
                    p["lease_id"] is None and p["status"] in TERMINAL | {"awaiting_input"}
                ):
                    p["error"] = code
                    obj = self._save(conn, principal, obj, p)
                    self._audit(conn, principal, obj, "provider_cleanup_failed", {"code": code})
                return obj
            if p["status"] in TERMINAL:
                conn.execute(
                    "UPDATE calls SET status='unknown',error_code='interrupted_call' "
                    "WHERE task_id=? AND owner=? AND lease_id=? AND status='started'",
                    (task_id, principal.user_id, lease_id),
                )
                p.update(lease_id=None, lease_until_ms=0)
                if code in {"provider_cleanup_unknown", "provider_unknown"}:
                    p["error"] = code
                if elapsed is not None and math.isfinite(elapsed):
                    p["elapsed_s"] = max(p["elapsed_s"], elapsed)
                return self._save(conn, principal, obj, p)
            conn.execute(
                "UPDATE calls SET status='unknown',error_code='interrupted_call' "
                "WHERE task_id=? AND owner=? AND lease_id=? AND status='started'",
                (task_id, principal.user_id, lease_id),
            )
            p["status"] = "blocked"
            p["error"] = code
            p["lease_id"] = None
            p["lease_until_ms"] = 0
            if elapsed is not None and math.isfinite(elapsed):
                p["elapsed_s"] = max(p["elapsed_s"], elapsed)
            obj = self._save(conn, principal, obj, p)
            self._audit(conn, principal, obj, "task_blocked", {"code": code})
            return obj

    @contextmanager
    def _provider_scope(self):
        session = getattr(self.provider, "session", None)
        if session is not None:
            with session():
                yield
        else:
            yield

    def run(self, principal, task_id):
        self.get(principal, task_id)
        if not self._provider_lock.acquire(blocking=False):
            raise DomainError("provider_busy", "A local model run is already active")
        try:
            return self._run(principal, task_id)
        finally:
            self._provider_lock.release()

    def _run(self, principal, task_id):
        lease_id = _identifier("lease-")
        started = self.monotonic()
        if not isinstance(started, (int, float)) or not math.isfinite(started):
            raise DomainError("clock_invalid", "Invalid monotonic clock")
        with self.store.transaction() as conn:
            obj = self._load(conn, principal, task_id)
            p = obj["payload"]
            if p["status"] in TERMINAL or p["status"] == "awaiting_input":
                if (p["status"] == "cancelled" and p["unknown_calls"]
                        and (p["lease_id"] is None or self._now() >= p["lease_until_ms"])):
                    conn.execute(
                        "UPDATE calls SET status='unknown',error_code='interrupted_call' "
                        "WHERE task_id=? AND owner=? AND status='started'",
                        (task_id, principal.user_id),
                    )
                    p.update(lease_id=None, lease_until_ms=0)
                    return self._save(conn, principal, obj, p)
                return obj
            if p["lease_id"] is not None and self._now() < p["lease_until_ms"]:
                raise DomainError("busy", "Task already has an active worker")
            if p["unknown_calls"]:
                conn.execute(
                    "UPDATE calls SET status='unknown',error_code='interrupted_call' "
                    "WHERE task_id=? AND owner=? AND status='started'",
                    (task_id, principal.user_id),
                )
                p.update(status="blocked", error="model_unknown", lease_id=None, lease_until_ms=0)
                return self._save(conn, principal, obj, p)
            self._snapshot(conn, principal, p)
            if p["provider_fingerprint"] != self.provider.fingerprint:
                raise DomainError("provider_changed", "Resume requires the original provider")
            self._budget(p)
            p.update(status="running", error=None, lease_id=lease_id,
                     lease_until_ms=self._now() + int(LEASE_SECONDS * 1000))
            base = p["elapsed_s"]
            previous_call_elapsed = p["accounted_call_elapsed_s"]
            self._save(conn, principal, obj, p)

        def runtime_elapsed(conn):
            previous = conn.execute(
                "SELECT COALESCE(SUM(elapsed_s),0) FROM calls "
                "WHERE task_id=? AND owner=? AND lease_id<>?",
                (task_id, principal.user_id, lease_id),
            ).fetchone()[0]
            return self._elapsed(started, base) + max(0.0, previous - previous_call_elapsed)

        def agent_node(state):
            with self.store.transaction() as conn:
                obj = self._load(conn, principal, task_id)
                self._fence(obj, lease_id)
                p = obj["payload"]
                p["elapsed_s"] = max(p["elapsed_s"], runtime_elapsed(conn))
                snapshot = self._snapshot(conn, principal, p)
                if p["pending_action"] is not None:
                    return {"stop": False}
                completed = conn.execute(
                    "SELECT call_id,response_hash FROM calls WHERE task_id=? AND owner=? "
                    "AND status='completed' ORDER BY created_at_ms,call_id",
                    (task_id, principal.user_id),
                ).fetchall()
                unhandled = [row for row in completed if row["call_id"] not in p["handled_call_ids"]]
                if unhandled:
                    if len(unhandled) != 1 or not unhandled[0]["response_hash"]:
                        raise DomainError("integrity", "Ambiguous unhandled model response")
                    response = self.store.get_artifact(unhandled[0]["response_hash"])
                    if set(response) != {"action"}:
                        raise DomainError("integrity", "Invalid stored model response")
                    p["pending_action"] = {
                        "call_id": unhandled[0]["call_id"], "action": response["action"],
                    }
                    self._save(conn, principal, obj, p)
                    return {"stop": False}
                menu = self._available(p, snapshot)
                if not menu:
                    p.update(status="no_recommendation", lease_id=None, lease_until_ms=0)
                    p["error"] = "abstained_or_no_safe_candidate"
                    self._save(conn, principal, obj, p)
                    self._audit(conn, principal, obj, "no_recommendation", {})
                    return {"stop": True}
                self._budget(p, call=True)
                model_context = self._model_context(task_id, p, snapshot)
                call_id = _identifier("call-")
                conn.execute(
                    "INSERT INTO calls(call_id,task_id,owner,lease_id,status,created_at_ms) "
                    "VALUES(?,?,?,?,'started',?)",
                    (call_id, task_id, principal.user_id, lease_id, self._now()),
                )
                self._save(conn, principal, obj, p)
            sent = False
            call_start = None
            tokens = None
            reply = None
            accounted = False
            try:
                call_start = self.monotonic()
                if not isinstance(call_start, (int, float)) or not math.isfinite(call_start):
                    raise DomainError("clock_invalid", "Invalid monotonic clock")
                with self.store.transaction() as conn:
                    current = self._load(conn, principal, task_id)
                    self._fence(current, lease_id)
                    current_p = current["payload"]
                    current_p["elapsed_s"] = max(current_p["elapsed_s"], runtime_elapsed(conn))
                    reserved = conn.execute(
                        "SELECT status FROM calls WHERE call_id=? AND task_id=? AND owner=? "
                        "AND lease_id=?", (call_id, task_id, principal.user_id, lease_id),
                    ).fetchone()
                    if reserved is None or reserved["status"] != "started":
                        raise DomainError("call_fence", "Reserved call was changed before dispatch")
                    self._budget({**current_p, "unknown_calls": current_p["unknown_calls"] - 1})
                    self._snapshot(conn, principal, current_p)
                    self._save(conn, principal, current, current_p)
                sent = True
                reply = self.provider.decide(model_context)
                if not isinstance(reply, ProviderReply):
                    raise DomainError("invalid_reply", "Provider returned an invalid reply")
                tokens = reply.tokens
                if type(tokens) is not int or not 0 <= tokens <= 100000:
                    raise DomainError("invalid_usage", "Provider returned invalid usage")
                elapsed = self._elapsed(call_start, 0.0)
                self._account(call_id, task_id, principal.user_id, lease_id, status="completed",
                              tokens=tokens, elapsed=elapsed, response={"action": reply.action})
                accounted = True
            except Exception as exc:
                if accounted:
                    raise
                if isinstance(exc, ProviderError):
                    tokens = exc.tokens
                    sent = exc.sent
                if not sent:
                    tokens = 0
                known = type(tokens) is int and 0 <= tokens <= 100000
                error = getattr(exc, "code", "provider_failed") if known else "model_unknown"
                try:
                    if call_start is None:
                        elapsed = 0.0
                    else:
                        elapsed = max(0.0, self.monotonic() - call_start)
                    if not math.isfinite(elapsed):
                        elapsed = 0.0
                except Exception:
                    elapsed = 0.0
                unknown = not known or (sent and error in {
                    "provider_unknown", "provider_cleanup_unknown", "model_unknown",
                })
                self._account(call_id, task_id, principal.user_id, lease_id,
                              status="unknown" if unknown else "failed", tokens=tokens if known else None,
                              elapsed=elapsed, error=error)
                raise DomainError(error, "Provider call did not produce a usable result") from None
            finally:
                with self.store.transaction() as conn:
                    row = conn.execute(
                        "SELECT status,tokens,response_hash,error_code FROM calls WHERE call_id=?",
                        (call_id,),
                    ).fetchone()
                    if row is None or row["status"] not in {"completed", "failed", "unknown"}:
                        raise DomainError("call_fence", "Model journal did not reach a recorded state")
            with self.store.transaction() as conn:
                obj = self._load(conn, principal, task_id)
                self._fence(obj, lease_id)
                p = obj["payload"]
                p["elapsed_s"] = max(p["elapsed_s"], runtime_elapsed(conn))
                self._budget(p)
                self._snapshot(conn, principal, p)
                p["pending_action"] = {"call_id": call_id, "action": reply.action}
                self._save(conn, principal, obj, p)
            return {"stop": False}

        def guard_node(state):
            with self.store.transaction() as conn:
                obj = self._load(conn, principal, task_id)
                self._fence(obj, lease_id)
                p = obj["payload"]
                p["elapsed_s"] = max(p["elapsed_s"], runtime_elapsed(conn))
                self._budget(p, tool=True)
                snapshot = self._snapshot(conn, principal, p)
                self._validate_action(self._pending_action(conn, principal, task_id, p), p, snapshot)
                self._save(conn, principal, obj, p)
            return {"stop": False}

        def tools_node(state):
            with self.store.transaction() as conn:
                obj = self._load(conn, principal, task_id)
                self._fence(obj, lease_id)
                p = copy.deepcopy(obj["payload"])
                p["elapsed_s"] = max(p["elapsed_s"], runtime_elapsed(conn))
                self._budget(p, tool=True)
                snapshot = self._snapshot(conn, principal, p)
                tool, args = self._validate_action(
                    self._pending_action(conn, principal, task_id, p), p, snapshot,
                )
                ctx = ProcessContext.model_validate(snapshot["context"])
                result = None
                if tool == "get_process_context":
                    result = {"context": snapshot["context"], "issues": validate_context(ctx, self._now())}
                elif tool == "retrieve_experience":
                    result = self.service.experience(principal, p["context_id"])
                elif tool == "predict_bc":
                    if snapshot["model"] is None:
                        result = {"status": "abstain", "raw_delta": None, "combination": [],
                                  "confidence": None, "issues": ["model_missing"],
                                  "model_hash": None, "context_hash": snapshot["context_hash"]}
                    else:
                        result = predict(snapshot["model"], ctx, self._now())
                    p["prediction"] = result
                elif tool == "draft_recommendation":
                    p["draft"] = {"raw_delta": p["prediction"]["raw_delta"],
                                  "model_hash": snapshot["model_hash"],
                                  "context_hash": snapshot["context_hash"]}
                elif tool == "check_constraints":
                    candidate = check_candidate(ctx, p["draft"]["raw_delta"], self._now())
                    candidate.update(task_id=task_id, context_id=p["context_id"],
                                     model_hash=snapshot["model_hash"])
                    candidate["candidate_hash"] = digest(candidate)
                    p["candidate"] = candidate
                    result = {"candidate": candidate}
                elif tool == "request_human_input":
                    p.update(status="awaiting_input", awaiting_fields=args["fields"],
                             question=args["question"], lease_id=None, lease_until_ms=0)
                elif tool == "submit_recommendation":
                    rec = self.service.submit_recommendation(
                        conn, principal, task_id=task_id, context_id=p["context_id"],
                        expected_snapshot=p["source_snapshot"], candidate=p["candidate"],
                        evidence=p["evidence"], explanation=args["explanation"],
                    )
                    p.update(status="submitted", recommendation_id=rec["object_id"],
                             lease_id=None, lease_until_ms=0)
                if result is not None:
                    p["evidence"].append(self._sign(task_id, p, tool, result))
                p["tools"] += 1
                p["handled_call_ids"].append(p["pending_action"]["call_id"])
                p["pending_action"] = None
                p["elapsed_s"] = max(p["elapsed_s"], runtime_elapsed(conn))
                if p["elapsed_s"] >= MAX_SECONDS:
                    raise DomainError("budget_exhausted", "Task budget exhausted")
                self._save(conn, principal, obj, p)
                self._audit(conn, principal, obj, "tool_completed", {"tool": tool})
                return {"stop": p["status"] != "running"}

        graph = StateGraph(GraphState)
        graph.add_node("agent", agent_node)
        graph.add_node("guard", guard_node)
        graph.add_node("tools", tools_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", lambda s: END if s["stop"] else "guard")
        graph.add_edge("guard", "tools")
        graph.add_conditional_edges("tools", lambda s: END if s["stop"] else "agent")
        try:
            with tracing_context(enabled=False), self._provider_scope():
                graph.compile().invoke({"task_id": task_id, "stop": False},
                                       config={"recursion_limit": 80, "callbacks": []})
        except Exception as exc:
            code = exc.code if isinstance(exc, DomainError) else "worker_failed"
            self._stop(principal, task_id, lease_id, code, runtime_elapsed)
        return self.get(principal, task_id)

    def cancel(self, principal, task_id, *, action_id):
        request = {"operation": "cancel", "task_id": task_id}
        with self.store.transaction() as conn:
            obj = self._load(conn, principal, task_id)
            old = self.store.receipt(conn, principal, action_id, request)
            if old is not None:
                return old
            p = obj["payload"]
            if p["status"] in TERMINAL:
                raise DomainError("terminal", "Terminal task cannot be cancelled")
            # Keep lease identity until an in-flight call accounts its reliable usage.
            p.update(status="cancelled", error="cancelled", pending_action=None)
            obj = self._save(conn, principal, obj, p)
            self._audit(conn, principal, obj, "task_cancelled", {})
            self.store.record_receipt(conn, principal, action_id, request, obj)
            return obj

    def resume(self, principal, task_id, *, action_id, fields, note):
        _note(note)
        request = {"operation": "supplement", "task_id": task_id, "fields": fields, "note": note}
        with self.store.transaction() as conn:
            obj = self._load(conn, principal, task_id)
            old = self.store.receipt(conn, principal, action_id, request)
            if old is not None:
                return old
            p = obj["payload"]
            if p["status"] != "awaiting_input":
                raise DomainError("state", "Task is not waiting for human input")
            if not isinstance(fields, dict) or sorted(fields) != sorted(p["awaiting_fields"]):
                raise DomainError("invalid_input", "Supply exactly the requested missing fields", 422)
            self._snapshot(conn, principal, p)
            self.service.supplement_context(
                conn, principal, p["context_id"], expected_hash=p["source_snapshot"]["context_hash"],
                fields=fields, note=note,
            )
            snapshot = self.service.snapshot(principal, p["context_id"], conn=conn)
            p.update(source_snapshot=self._sources(snapshot), evidence=[], prediction=None,
                     candidate=None, draft=None, pending_action=None, awaiting_fields=[],
                     status="created", error=None, lease_id=None, lease_until_ms=0)
            obj = self._save(conn, principal, obj, p)
            self._audit(conn, principal, obj, "human_supplement", {"fields": sorted(fields), "note": note})
            self.store.record_receipt(conn, principal, action_id, request, obj)
            return obj

    def reconcile(self, principal, task_id, *, action_id, call_id, no_result, tokens, note):
        if no_result is not True:
            raise DomainError("attestation_required", "Explicit no-result attestation required", 422)
        if tokens is not None and (type(tokens) is not int or not 0 <= tokens <= 100000):
            raise DomainError("invalid_usage", "Invalid token accounting", 422)
        _note(note)
        request = {"operation": "reconcile", "task_id": task_id, "call_id": call_id,
                   "no_result": no_result, "tokens": tokens, "note": note}
        with self.store.transaction() as conn:
            obj = self._load(conn, principal, task_id)
            self.store.require(principal, obj["plant_id"], {"admin"}, conn=conn)
            old = self.store.receipt(conn, principal, action_id, request)
            if old is not None:
                return old
            p = obj["payload"]
            if p["lease_id"] is not None and self._now() < p["lease_until_ms"]:
                raise DomainError("busy", "Cannot reconcile an active worker")
            row = conn.execute("SELECT * FROM calls WHERE call_id=? AND task_id=? AND owner=?",
                               (call_id, task_id, principal.user_id)).fetchone()
            if row is not None and row["status"] == "started":
                conn.execute(
                    "UPDATE calls SET status='unknown',error_code='interrupted_call' "
                    "WHERE call_id=? AND status='started'", (call_id,),
                )
            elif row is None or row["status"] != "unknown":
                raise DomainError("state", "Call is not awaiting reconciliation")
            known = row["tokens"] or 0
            charge = known + max(0, MAX_TOKENS - p["tokens"]) if tokens is None else tokens
            if charge < known:
                raise DomainError("invalid_usage", "Known usage cannot be decreased")
            conn.execute("UPDATE calls SET status='failed',tokens=?,error_code='reconciled_no_result' "
                         "WHERE call_id=? AND status='unknown'", (charge, call_id))
            p.update(status="cancelled" if p["status"] == "cancelled" else "created",
                     pending_action=None, error=None, lease_id=None, lease_until_ms=0)
            usage = self._usage(conn, task_id, principal.user_id)
            p.update(calls=usage["calls"], tokens=usage["tokens"], unknown_calls=usage["unknown_calls"])
            obj = self._save(conn, principal, obj, p)
            self._audit(conn, principal, obj, "call_reconciled", {"call_id": call_id,
                        "tokens": charge, "no_result": True, "note": note})
            self.store.record_receipt(conn, principal, action_id, request, obj)
            return obj
