"""Human-gated actions on a local synthetic device, never a production adapter.

Approval, an execution request, an adapter acknowledgement, a verified readback,
and mature simulated observations are deliberately different facts. SQLite only
provides atomicity for this local simulator, not exactly-once physical actuation.
"""

from __future__ import annotations

import math
import re
import time
from typing import TYPE_CHECKING

from .domain import ProcessContext, check_candidate, digest
from .errors import DomainError
from .fixtures import simulate_quality

if TYPE_CHECKING:
    from .service import RecommendationService

_SOURCE_KEYS = (
    "context_hash",
    "context_version",
    "model_hash",
    "model_version",
    "model_content_hash",
    "constraint_version",
)
_REQUEST_KEYS = {
    "execution_id",
    "request_id",
    "recommendation_id",
    "recommendation_hash",
    "recommendation_version",
    "owner",
    "plant_id",
    "machine_id",
    "source",
    "source_hash",
    "candidate_hash",
    "before",
    "target",
    "expected_device_version",
    "expected_device_revision",
    "requested_at_ms",
    "request_action_id",
    "synthetic",
}
_MIN_OBSERVATION_AGE_MS = 60_000


def _identifier(value):
    if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise DomainError("invalid_identifier", "A bounded identifier is required", 422)
    return value


def _hash(value):
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise DomainError("invalid_hash", "An exact SHA256 content hash is required", 422)
    return value


def _same(actual, expected, message="Stored action evidence does not match"):
    # JSON equality, unlike Python equality, distinguishes True, 1 and 1.0.
    if digest(actual) != digest(expected):
        raise DomainError("action_integrity", message)


def _integer(value, minimum=0):
    return type(value) is int and minimum <= value <= 2**63 - 1


def _state(obj, *allowed):
    if obj["status"] not in allowed:
        raise DomainError("invalid_state", "This action is not allowed in the current state")


def _wilson(count, total):
    z = 1.959963984540054
    rate = count / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total**2)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


class ActionService:
    def __init__(self, service: RecommendationService, *, clock=None):
        self.service = service
        self.store = service.store
        self.clock = clock if clock is not None else lambda: time.time_ns() // 1_000_000

    def _now(self):
        now = self.clock()
        if not _integer(now):
            raise DomainError(
                "invalid_time", "Clock must return nonnegative integer milliseconds", 422
            )
        return now

    def _start(self, conn, principal, kind, object_id, action_id, request, roles=None):
        _identifier(object_id)
        _identifier(action_id)
        obj = self.store.get_object(kind, object_id, principal, conn=conn)
        self.store.require(principal, obj["plant_id"], roles=roles, conn=conn)
        replay = self.store.receipt(conn, principal, action_id, request)
        return obj, replay

    def _update(self, conn, principal, obj, status, payload):
        return self.store.update_object(
            conn,
            kind=obj["kind"],
            object_id=obj["object_id"],
            principal=principal,
            expected_version=obj["version"],
            status=status,
            payload=payload,
        )

    def _finish(self, conn, principal, action_id, request, result, action, detail):
        self.store.audit(
            conn,
            principal,
            kind=result["kind"],
            object_id=result["object_id"],
            action=action,
            detail={"synthetic": True, **detail},
        )
        self.store.record_receipt(conn, principal, action_id, request, result)
        return result

    def _source(self, conn, principal, recommendation, now):
        payload = recommendation["payload"]
        if payload.get("synthetic") is not True:
            raise DomainError("action_integrity", "Only synthetic recommendations may be executed")
        snapshot = self.service.snapshot(principal, payload.get("context_id"), conn=conn)
        if snapshot.get("model") is None:
            raise DomainError("model_required", "Execution requires the bound trained model")
        for key in _SOURCE_KEYS:
            if key not in payload or key not in snapshot:
                raise DomainError("action_integrity", "Recommendation source binding is incomplete")
            _same(
                payload[key],
                snapshot[key],
                "Recommendation source changed; create a new recommendation",
            )
        if "experience_version" in payload:
            _same(payload["experience_version"], snapshot.get("experience_version"))
        try:
            context = ProcessContext.model_validate(snapshot["context"])
        except (ValueError, TypeError, KeyError) as exc:
            raise DomainError("action_integrity", "Stored source context is invalid") from exc
        _same(context.context_id, payload["context_id"])
        _same(context.plant_id, recommendation["plant_id"])
        candidate = payload.get("candidate")
        if type(candidate) is not dict or "candidate_hash" not in candidate:
            raise DomainError("action_integrity", "Recommendation lacks a frozen candidate")
        _same(
            candidate["candidate_hash"],
            digest({k: v for k, v in candidate.items() if k != "candidate_hash"}),
        )
        for key in ("task_id", "context_id", "model_hash"):
            if key not in payload:
                raise DomainError(
                    "action_integrity", "Recommendation candidate binding is incomplete"
                )
            _same(candidate.get(key), payload[key])
        checked = check_candidate(context, candidate.get("raw_delta"), now)
        if checked["eligible"] is not True:
            raise DomainError(
                "unsafe_candidate", "Candidate is stale or fails current hard constraints"
            )
        for key, value in checked.items():
            _same(
                candidate.get(key),
                value,
                "Frozen candidate differs from independently checked constraints",
            )
        source = {key: payload[key] for key in _SOURCE_KEYS}
        source.update(
            context_id=context.context_id,
            plant_id=context.plant_id,
            machine_id=context.machine_id,
            process_version=context.process_version,
        )
        return context, candidate, source

    def _device(self, conn, principal, source, current, *, version=None, revision=None):
        device = self.store.get_object("device", source["machine_id"], principal, conn=conn)
        payload = device["payload"]
        if payload.get("synthetic") is not True:
            raise DomainError("action_integrity", "A synthetic device is required")
        for key in ("machine_id", "plant_id", "process_version", "context_id", "context_hash"):
            _same(
                payload.get(key),
                source[key],
                "Device active context does not match the recommendation",
            )
        _same(device["plant_id"], source["plant_id"])
        _same(
            payload.get("current"), current, "Device parameters no longer match the frozen action"
        )
        if not _integer(payload.get("revision"), 1) or not _integer(payload.get("updated_at_ms")):
            raise DomainError("action_integrity", "Device revision or timestamp is invalid")
        if version is not None:
            _same(device["version"], version, "Device object version changed")
        if revision is not None:
            _same(payload["revision"], revision, "Device revision changed")
        return device

    @staticmethod
    def _approval(recommendation):
        payload = recommendation["payload"]
        approval = payload.get("approval")
        if (
            type(approval) is not dict
            or approval.get("decision") != "approved"
            or approval.get("synthetic") is not True
        ):
            raise DomainError("action_integrity", "An exact synthetic approval is required")
        original = {
            k: v
            for k, v in payload.items()
            if k not in {"approval", "execution", "readback", "observation"}
        }
        _same(
            approval.get("recommendation_hash"),
            digest(original),
            "Approval does not bind the original recommendation",
        )

    def review(
        self, principal, recommendation_id, *, action_id, recommendation_hash, decision, note
    ):
        _hash(recommendation_hash)
        if type(decision) is not str or decision not in {"approved", "rejected"}:
            raise DomainError("invalid_decision", "Decision must be approved or rejected", 422)
        if type(note) is not str or not note.strip() or len(note) > 4000:
            raise DomainError(
                "invalid_note", "A nonempty review note of at most 4000 characters is required", 422
            )
        request = dict(
            operation="review",
            recommendation_id=recommendation_id,
            recommendation_hash=recommendation_hash,
            decision=decision,
            note=note,
        )
        with self.store.transaction() as conn:
            recommendation, replay = self._start(
                conn,
                principal,
                "recommendation",
                recommendation_id,
                action_id,
                request,
                roles=("reviewer", "admin"),
            )
            if replay is not None:
                return replay
            _state(recommendation, "pending_review")
            _same(
                recommendation["content_hash"],
                recommendation_hash,
                "Review requires the exact recommendation hash",
            )
            if recommendation["payload"].get("synthetic") is not True:
                raise DomainError(
                    "action_integrity", "Only synthetic recommendations may be reviewed"
                )
            approval = dict(
                decision=decision,
                recommendation_hash=recommendation_hash,
                user_id=principal.user_id,
                reviewed_at_ms=self._now(),
                note=note,
                synthetic=True,
                notice="Review of a synthetic recommendation; no device action.",
            )
            result = self._update(
                conn,
                principal,
                recommendation,
                decision,
                {**recommendation["payload"], "approval": approval},
            )
            return self._finish(conn, principal, action_id, request, result, "review", approval)

    def request_execution(
        self, principal, recommendation_id, *, action_id, recommendation_hash, confirmed: bool
    ):
        _hash(recommendation_hash)
        if confirmed is not True:
            raise DomainError(
                "confirmation_required",
                "Explicit confirmed=True is required for synthetic execution",
                422,
            )
        request = dict(
            operation="request_execution",
            recommendation_id=recommendation_id,
            recommendation_hash=recommendation_hash,
            confirmed=confirmed,
        )
        execution_id = _identifier("exec-" + _identifier(recommendation_id))
        with self.store.transaction() as conn:
            recommendation, replay = self._start(
                conn, principal, "recommendation", recommendation_id, action_id, request
            )
            if replay is not None:
                return replay
            _state(recommendation, "approved")
            _same(
                recommendation["content_hash"],
                recommendation_hash,
                "Execution requires the exact approved recommendation hash",
            )
            self._approval(recommendation)
            now = self._now()
            context, candidate, source = self._source(conn, principal, recommendation, now)
            before = context.current.model_dump(mode="json")
            device = self._device(conn, principal, source, before)
            if device["payload"]["updated_at_ms"] > now:
                raise DomainError("action_integrity", "Device timestamp is in the future")
            fields = dict(
                execution_id=execution_id,
                request_id=execution_id,
                recommendation_id=recommendation_id,
                recommendation_hash=recommendation_hash,
                recommendation_version=recommendation["version"],
                owner=principal.user_id,
                plant_id=context.plant_id,
                machine_id=context.machine_id,
                source=source,
                source_hash=digest(source),
                candidate_hash=candidate["candidate_hash"],
                before=before,
                target=candidate["target"],
                expected_device_version=device["version"],
                expected_device_revision=device["payload"]["revision"],
                requested_at_ms=now,
                request_action_id=action_id,
                synthetic=True,
            )
            device_request = {**fields, "confirmed": True}
            execution = self.store.create_object(
                conn,
                kind="execution",
                object_id=execution_id,
                principal=principal,
                plant_id=context.plant_id,
                status="pending",
                payload={
                    **fields,
                    "request": device_request,
                    "request_hash": digest(device_request),
                },
            )
            pending = self._update(
                conn,
                principal,
                recommendation,
                "execution_pending",
                {
                    **recommendation["payload"],
                    "execution": {
                        "execution_id": execution_id,
                        "request_id": execution_id,
                        "request_hash": execution["payload"]["request_hash"],
                        "recommendation_hash": recommendation_hash,
                        "requested_at_ms": now,
                    },
                },
            )
            self.store.audit(
                conn,
                principal,
                kind="recommendation",
                object_id=pending["object_id"],
                action="execution_requested",
                detail={"execution_id": execution_id, "synthetic": True},
            )
            return self._finish(
                conn,
                principal,
                action_id,
                request,
                execution,
                "request_execution",
                {
                    "recommendation_id": recommendation_id,
                    "request_hash": execution["payload"]["request_hash"],
                },
            )

    def _execution(self, conn, principal, execution):
        payload = execution["payload"]
        request = payload.get("request")
        if (
            type(request) is not dict
            or set(request) != _REQUEST_KEYS | {"confirmed"}
            or request.get("confirmed") is not True
        ):
            raise DomainError("action_integrity", "Execution request binding is invalid")
        _same(payload.get("request_hash"), digest(request))
        for key in _REQUEST_KEYS:
            _same(payload.get(key), request[key])
        if payload.get("synthetic") is not True:
            raise DomainError("action_integrity", "Only synthetic device requests are accepted")
        _same(payload["execution_id"], execution["object_id"])
        _same(payload["request_id"], execution["object_id"])
        _same(payload["owner"], principal.user_id)
        _same(payload["plant_id"], execution["plant_id"])
        _same(payload["source_hash"], digest(payload["source"]))
        for key in (
            "expected_device_version",
            "expected_device_revision",
            "recommendation_version",
        ):
            if not _integer(payload[key], 1):
                raise DomainError("action_integrity", "Execution revision binding is invalid")
        if not _integer(payload["requested_at_ms"]):
            raise DomainError("action_integrity", "Execution timestamp is invalid")
        recommendation = self.store.get_object(
            "recommendation", payload["recommendation_id"], principal, conn=conn
        )
        _same(recommendation["plant_id"], payload["plant_id"])
        self._approval(recommendation)
        approved = {
            k: v
            for k, v in recommendation["payload"].items()
            if k not in {"execution", "readback", "observation"}
        }
        _same(digest(approved), payload["recommendation_hash"])
        _same(
            recommendation["payload"].get("execution"),
            {
                "execution_id": payload["execution_id"],
                "request_id": payload["request_id"],
                "request_hash": payload["request_hash"],
                "recommendation_hash": payload["recommendation_hash"],
                "requested_at_ms": payload["requested_at_ms"],
            },
        )
        return recommendation

    @staticmethod
    def _receipt_fields(payload, now):
        return {
            **{
                key: payload[key]
                for key in (
                    "execution_id",
                    "request_id",
                    "request_hash",
                    "recommendation_id",
                    "recommendation_hash",
                    "owner",
                    "plant_id",
                    "machine_id",
                    "source",
                    "source_hash",
                    "candidate_hash",
                    "before",
                    "target",
                    "synthetic",
                )
            },
            "device_version": payload["expected_device_version"] + 1,
            "device_revision": payload["expected_device_revision"] + 1,
            "applied_at_ms": now,
        }

    def apply_synthetic(self, principal, execution_id, *, action_id, fault: str | None = None):
        if fault is not None and (type(fault) is not str or fault != "after_apply"):
            raise DomainError("invalid_fault", "Unsupported synthetic adapter fault", 422)
        request = dict(operation="apply_synthetic", execution_id=execution_id, fault=fault)
        # Device mutation and its receipt commit with a conservative unknown
        # state. A separately committed ACK is optional, never a prerequisite to
        # recovery. Reserve the global action_id in that first transaction too.
        with self.store.transaction() as conn:
            execution, replay = self._start(
                conn, principal, "execution", execution_id, action_id, request
            )
            if replay is not None:
                return self._apply_replay(conn, principal, execution, action_id, request, replay)
            _state(execution, "pending")
            recommendation = self._execution(conn, principal, execution)
            _state(recommendation, "execution_pending")
            now = self._now()
            payload = execution["payload"]
            context, candidate, source = self._source(conn, principal, recommendation, now)
            _same(source, payload["source"])
            _same(candidate["target"], payload["target"])
            _same(candidate["candidate_hash"], payload["candidate_hash"])
            _same(context.current.model_dump(mode="json"), payload["before"])
            if now < payload["requested_at_ms"]:
                raise DomainError("invalid_time", "Clock precedes the execution request")
            device = self._device(
                conn,
                principal,
                source,
                payload["before"],
                version=payload["expected_device_version"],
                revision=payload["expected_device_revision"],
            )
            if now < device["payload"]["updated_at_ms"]:
                raise DomainError("invalid_time", "Clock precedes the device state")
            device = self._update(
                conn,
                principal,
                device,
                device["status"],
                {
                    **device["payload"],
                    "current": payload["target"],
                    "revision": payload["expected_device_revision"] + 1,
                    "updated_at_ms": now,
                },
            )
            receipt = self.store.create_object(
                conn,
                kind="device_receipt",
                object_id=execution_id,
                principal=principal,
                plant_id=execution["plant_id"],
                status="applied",
                payload=self._receipt_fields(payload, now),
            )
            result = self._update(
                conn,
                principal,
                execution,
                "unknown",
                {
                    **payload,
                    "device_receipt_hash": receipt["content_hash"],
                    "applied_at_ms": now,
                    "adapter_acknowledged": False,
                    "adapter_outcome": "acknowledgement_unconfirmed",
                },
            )
            self._update(conn, principal, recommendation, "unknown", recommendation["payload"])
            self.store.audit(
                conn,
                principal,
                kind="device",
                object_id=device["object_id"],
                action="synthetic_apply",
                detail={
                    "execution_id": execution_id,
                    "device_receipt_hash": receipt["content_hash"],
                    "synthetic": True,
                },
            )
            self.store.audit(
                conn,
                principal,
                kind="execution",
                object_id=execution_id,
                action="apply_synthetic",
                detail={
                    "outcome": "unknown",
                    "device_receipt_hash": receipt["content_hash"],
                    "notice": "Local synthetic transaction only; not physical exactly-once delivery.",
                    "synthetic": True,
                },
            )
            # Store receipts are immutable. An authorized result row is the
            # indirection for optional ACK finalization, not a resend token.
            result_id = "apply-result-" + digest(
                {"owner": principal.user_id, "action_id": action_id}
            )
            reservation = {
                "operation": "apply_synthetic_result",
                "execution_id": execution_id,
                "result_id": result_id,
                "request_hash": digest(request),
            }
            self.store.create_object(
                conn,
                kind="action_result",
                object_id=result_id,
                principal=principal,
                plant_id=execution["plant_id"],
                status="unknown",
                payload={"reservation": reservation, "response": result},
            )
            self.store.record_receipt(conn, principal, action_id, request, reservation)
        # The device transaction is closed before either this fault or ACK.
        if fault == "after_apply":
            return result
        return self._acknowledge(principal, execution_id, action_id, request, reservation)

    def _apply_replay(self, conn, principal, execution, action_id, request, reservation):
        expected = {
            "operation": "apply_synthetic_result",
            "execution_id": execution["object_id"],
            "result_id": "apply-result-"
            + digest({"owner": principal.user_id, "action_id": action_id}),
            "request_hash": digest(request),
        }
        _same(reservation, expected)
        stored = self.store.get_object("action_result", expected["result_id"], principal, conn=conn)
        _same(stored["plant_id"], execution["plant_id"])
        _same(stored["payload"].get("reservation"), expected)
        response = stored["payload"].get("response")
        if type(response) is not dict or type(response.get("payload")) is not dict:
            raise DomainError("action_integrity", "Stored adapter response is invalid")
        for key in ("object_id", "kind", "owner", "plant_id"):
            _same(response.get(key), execution[key])
        _same(response.get("content_hash"), digest(response["payload"]))
        _same(response["payload"].get("request_hash"), execution["payload"]["request_hash"])
        return response

    def _acknowledge(self, principal, execution_id, action_id, request, reservation):
        """Local adapter ACK only; it cannot claim verified parameter readback."""
        with self.store.transaction() as conn:
            execution, replay = self._start(
                conn, principal, "execution", execution_id, action_id, request
            )
            _same(replay, reservation)
            saved = self._apply_replay(conn, principal, execution, action_id, request, replay)
            result_obj = self.store.get_object(
                "action_result", reservation["result_id"], principal, conn=conn
            )
            # A concurrent readback can have already reconciled the unknown
            # request. Never overwrite its later fact or observation.
            if execution["status"] != "unknown" or execution["version"] != saved["version"]:
                return saved
            recommendation = self._execution(conn, principal, execution)
            _state(recommendation, "unknown")
            self._evidence(conn, principal, execution, self._now())
            result = self._update(
                conn,
                principal,
                execution,
                "applied",
                {
                    **execution["payload"],
                    "adapter_acknowledged": True,
                    "adapter_outcome": "synthetic_applied_not_read_back",
                },
            )
            self._update(
                conn, principal, recommendation, "execution_pending", recommendation["payload"]
            )
            self._update(
                conn,
                principal,
                result_obj,
                "applied",
                {
                    "reservation": reservation,
                    "response": result,
                },
            )
            self.store.audit(
                conn,
                principal,
                kind="execution",
                object_id=execution_id,
                action="synthetic_acknowledged",
                detail={
                    "synthetic": True,
                    "device_receipt_hash": result["payload"]["device_receipt_hash"],
                },
            )
            return result

    def _evidence(self, conn, principal, execution, now):
        payload = execution["payload"]
        receipt = self.store.get_object(
            "device_receipt", execution["object_id"], principal, conn=conn
        )
        _state(receipt, "applied")
        applied_at = receipt["payload"].get("applied_at_ms")
        if not _integer(applied_at) or not payload["requested_at_ms"] <= applied_at <= now:
            raise DomainError("action_integrity", "Device receipt timestamp is inconsistent")
        _same(receipt["payload"], self._receipt_fields(payload, applied_at))
        _same(receipt["plant_id"], execution["plant_id"])
        _same(receipt["content_hash"], payload.get("device_receipt_hash"))
        _same(applied_at, payload.get("applied_at_ms"))
        device = self._device(
            conn,
            principal,
            payload["source"],
            payload["target"],
            version=payload["expected_device_version"] + 1,
            revision=payload["expected_device_revision"] + 1,
        )
        _same(device["payload"]["updated_at_ms"], applied_at)
        return receipt, device

    def readback(self, principal, execution_id, *, action_id):
        request = dict(operation="readback", execution_id=execution_id)
        with self.store.transaction() as conn:
            execution, replay = self._start(
                conn, principal, "execution", execution_id, action_id, request
            )
            if replay is not None:
                return replay
            _state(execution, "applied", "unknown")
            recommendation = self._execution(conn, principal, execution)
            _state(recommendation, "execution_pending", "unknown")
            now = self._now()
            try:
                receipt, device = self._evidence(conn, principal, execution, now)
            except DomainError as exc:
                if exc.status not in {404, 409}:
                    raise
                # Human assertions never enter this API. Missing, damaged or
                # nonmatching stored evidence is not evidence of success.
                payload = {
                    **execution["payload"],
                    "readback_attempt": {
                        "at_ms": now,
                        "verified": False,
                        "reason": "matching_device_evidence_unavailable",
                    },
                }
                result = self._update(conn, principal, execution, "unknown", payload)
                self._update(conn, principal, recommendation, "unknown", recommendation["payload"])
                return self._finish(
                    conn,
                    principal,
                    action_id,
                    request,
                    result,
                    "readback_unknown",
                    payload["readback_attempt"],
                )
            readback = {
                "execution_id": execution_id,
                "request_id": execution["payload"]["request_id"],
                "request_hash": execution["payload"]["request_hash"],
                "source_hash": execution["payload"]["source_hash"],
                "device_receipt_hash": receipt["content_hash"],
                "device_hash": device["content_hash"],
                "device_version": device["version"],
                "device_revision": device["payload"]["revision"],
                "current": device["payload"]["current"],
                "readback_at_ms": now,
                "synthetic": True,
            }
            readback["readback_hash"] = digest(readback)
            result = self._update(
                conn,
                principal,
                execution,
                "executed",
                {**execution["payload"], "readback": readback},
            )
            updated = self._update(
                conn,
                principal,
                recommendation,
                "executed",
                {**recommendation["payload"], "readback": readback},
            )
            self.store.audit(
                conn,
                principal,
                kind="recommendation",
                object_id=updated["object_id"],
                action="readback_verified",
                detail={"readback_hash": readback["readback_hash"], "synthetic": True},
            )
            return self._finish(
                conn, principal, action_id, request, result, "readback_verified", readback
            )

    def observe(
        self, principal, recommendation_id, *, action_id, seed: int = 27183, sample_count: int = 120
    ):
        if not _integer(seed):
            raise DomainError("invalid_seed", "Seed must be a bounded nonnegative integer", 422)
        if type(sample_count) is not int or not 60 <= sample_count <= 10_000:
            raise DomainError(
                "invalid_sample_count", "Observation requires 60..10000 new samples per arm", 422
            )
        request = dict(
            operation="observe",
            recommendation_id=recommendation_id,
            seed=seed,
            sample_count=sample_count,
        )
        with self.store.transaction() as conn:
            recommendation, replay = self._start(
                conn, principal, "recommendation", recommendation_id, action_id, request
            )
            if replay is not None:
                return replay
            _state(recommendation, "executed")
            execution_id = recommendation["payload"].get("execution", {}).get("execution_id")
            execution = self.store.get_object("execution", execution_id, principal, conn=conn)
            _state(execution, "executed")
            bound_recommendation = self._execution(conn, principal, execution)
            _same(bound_recommendation["object_id"], recommendation_id)
            readback = execution["payload"].get("readback")
            if type(readback) is not dict or not _integer(readback.get("readback_at_ms")):
                raise DomainError("action_integrity", "A verified readback timestamp is required")
            _same(
                readback.get("readback_hash"),
                digest({k: v for k, v in readback.items() if k != "readback_hash"}),
            )
            _same(recommendation["payload"].get("readback"), readback)
            now = self._now()
            age = now - readback["readback_at_ms"]
            if age < _MIN_OBSERVATION_AGE_MS:
                raise DomainError(
                    "observation_immature",
                    "At least 60 real clock seconds must elapse after verified readback",
                )
            receipt, device = self._evidence(conn, principal, execution, now)
            if readback["readback_at_ms"] < receipt["payload"]["applied_at_ms"]:
                raise DomainError(
                    "action_integrity", "Readback cannot precede actual device application"
                )
            expected_readback = {
                "execution_id": execution_id,
                "request_id": execution["payload"]["request_id"],
                "request_hash": execution["payload"]["request_hash"],
                "source_hash": execution["payload"]["source_hash"],
                "device_receipt_hash": receipt["content_hash"],
                "device_hash": device["content_hash"],
                "device_version": device["version"],
                "device_revision": device["payload"]["revision"],
                "current": device["payload"]["current"],
                "readback_at_ms": readback["readback_at_ms"],
                "synthetic": True,
            }
            expected_readback["readback_hash"] = digest(expected_readback)
            _same(readback, expected_readback)
            source = execution["payload"]["source"]
            context_obj = self.service.get_context(principal, source["context_id"], conn=conn)
            _same(context_obj["version"], source["context_version"])
            _same(digest(context_obj["payload"]), source["context_hash"])
            try:
                context = ProcessContext.model_validate(context_obj["payload"])
            except (ValueError, TypeError) as exc:
                raise DomainError(
                    "action_integrity", "Observation source context is invalid"
                ) from exc
            before = simulate_quality(context, execution["payload"]["before"], seed, sample_count)
            after_seed = (seed + 1) % (2**63)
            after = simulate_quality(
                context, execution["payload"]["target"], after_seed, sample_count
            )
            for arm in (before, after):
                arm["rate_ci95"] = _wilson(arm["defect_count"], arm["sample_count"])
            difference = after["defect_rate"] - before["defect_rate"]
            low = difference - math.hypot(
                after["defect_rate"] - after["rate_ci95"][0],
                before["rate_ci95"][1] - before["defect_rate"],
            )
            high = difference + math.hypot(
                after["rate_ci95"][1] - after["defect_rate"],
                before["defect_rate"] - before["rate_ci95"][0],
            )
            conclusion = (
                "synthetic_observed_improvement"
                if high < 0
                else "worsened"
                if low > 0
                else "no_clear_improvement"
            )
            observation = {
                "synthetic": True,
                "observed_at_ms": now,
                "age_after_readback_ms": age,
                "readback_hash": readback["readback_hash"],
                "before_seed": seed,
                "after_seed": after_seed,
                "before": before,
                "after": after,
                "difference_after_minus_before": difference,
                "difference_ci95": [max(-1.0, low), min(1.0, high)],
                "interval_method": "Wilson marginal / Newcombe independent-binomial difference",
                "conclusion": conclusion,
                "causal_effect_proven": False,
                "paired_simulated_comparison": True,
                "independent_samples": True,
                "baseline_is_observed_production": False,
                "notice": "Independent synthetic before/after samples for the same frozen context; not an observed production baseline or field causal evidence.",
            }
            result = self._update(
                conn,
                principal,
                recommendation,
                "observed",
                {**recommendation["payload"], "observation": observation},
            )
            return self._finish(
                conn, principal, action_id, request, result, "synthetic_observation", observation
            )
