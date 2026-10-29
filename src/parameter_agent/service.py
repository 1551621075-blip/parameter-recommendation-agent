"""Owner-scoped sources and frozen recommendations for the synthetic example.

The eight experience cards below are original teaching notes derived from this
project's example BOP and simulator design. They are not retrieved production
records, fitted process knowledge, or evidence of a real-world improvement.
No model calls, training, approval, or equipment execution belong in this module.
"""
from __future__ import annotations

import math
import re
import sqlite3
import time
from collections.abc import Callable

from . import bc
from .domain import (
    CONSTRAINT_VERSION,
    ProcessContext,
    canonical,
    check_candidate,
    digest,
    validate_context,
)
from .errors import DomainError
from .storage import Principal, Store

EXPERIENCE_VERSION = "synthetic-experience-v1"
SOURCE_KEYS = (
    "context_hash", "context_version", "model_hash", "model_content_hash",
    "model_version", "experience_version", "constraint_version",
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_PROVENANCE = "Original synthetic teaching card; example BOP/simulator, not production evidence."
# Exact metadata matching is intentional: unfamiliar materials/processes must not
# inherit these notes. Two distinct notes per defect give eight original cards.
_EXPERIENCE = (
    ("synthetic-exp-fill-material", "short_shot",
     "In the synthetic resin-A example, viscosity changes the toy filling optimum. "
     "Record viscosity explicitly; do not infer a missing value from the defect label."),
    ("synthetic-exp-fill-limits", "short_shot",
     "The example short-shot BOP permits only nonnegative temperature, speed, or pressure "
     "changes, at most two axes. Experience cannot override raw-step or coupling checks."),
    ("synthetic-exp-flash-load", "flash",
     "The original flash simulator penalizes excess speed and pressure relative to a "
     "viscosity-dependent optimum. This is a teaching hypothesis, not a production diagnosis."),
    ("synthetic-exp-flash-direction", "flash",
     "The example flash BOP permits decreasing speed or pressure only. A plausible narrative "
     "is not permission to increase another parameter or bypass the current grid."),
    ("synthetic-exp-warp-balance", "warpage",
     "The synthetic warpage simulator couples its preferred cooling duration to ambient "
     "temperature. An absent ambient measurement needs explicit human supplementation."),
    ("synthetic-exp-warp-coupling", "warpage",
     "The example warpage BOP allows temperature reduction and cooling increase. The hot "
     "temperature/minimum cooling constraint still applies to raw and quantized targets."),
    ("synthetic-exp-normal-hold", "normal",
     "The example normal-process BOP allows only holding parameters. A zero change is not "
     "an executable recommendation and should not be described as corrective action."),
    ("synthetic-exp-normal-observe", "normal",
     "Independent synthetic quality samples can fluctuate even at unchanged setpoints. "
     "A defect-count difference alone is not proof of a causal production improvement."),
)


def _identifier(value: str, *, maximum: int = 80) -> str:
    if type(value) is not str or len(value) > maximum or not _IDENTIFIER.fullmatch(value):
        raise DomainError("validation", "Invalid source identity", 422)
    return value


def _model_id(principal: Principal, plant_id: str) -> str:
    # Store object IDs are global. Bind both identities without concatenation
    # ambiguity, retaining the full SHA256 in a bounded 77-character Identifier.
    binding = {"owner": _identifier(principal.user_id), "plant_id": _identifier(plant_id)}
    return "active-model-" + digest(binding)


def _text(value: str, *, maximum: int = 8000) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise DomainError("validation", "A bounded nonempty explanation is required", 422)
    canonical(value)
    return value


def _integrity() -> DomainError:
    return DomainError("integrity", "Stored source data failed integrity validation", 409)


def _same(left, right) -> bool:
    # Python equality would treat True, 1, and 1.0 as interchangeable.
    return canonical(left) == canonical(right)


class RecommendationService:
    def __init__(self, store: Store, *, clock: Callable[[], int] | None = None):
        self.store = store
        self.clock = clock if clock is not None else lambda: time.time_ns() // 1_000_000

    def _now(self) -> int:
        now = self.clock()
        if type(now) is not int or not 0 <= now <= 2**63 - 1:
            raise DomainError("invalid_time", "Invalid service clock", 422)
        return now

    def _optional(self, kind, object_id, principal, *, conn):
        try:
            return self.store.get_object(kind, object_id, principal, conn=conn)
        except DomainError as exc:
            if exc.code == "not_found" and exc.status == 404:
                return None
            raise

    @staticmethod
    def _context(obj: dict) -> ProcessContext:
        try:
            context = ProcessContext.model_validate(obj["payload"])
            if (context.context_id != obj["object_id"] or context.plant_id != obj["plant_id"]
                    or obj["kind"] != "context" or obj["status"] != "active"
                    or not _same(context.model_dump(mode="json"), obj["payload"])
                    or digest(obj["payload"]) != obj["content_hash"]):
                raise ValueError
            return context
        except (ValueError, TypeError, KeyError, DomainError) as exc:
            raise _integrity() from exc

    def get_context(self, principal: Principal, context_id: str, *, conn=None) -> dict:
        obj = self.store.get_object("context", context_id, principal, conn=conn)
        self._context(obj)
        return obj

    def get_model(self, principal: Principal, plant_id: str, *, conn=None) -> dict | None:
        if conn is None:
            with self.store.connect() as current:
                return self.get_model(principal, plant_id, conn=current)
        self.store.require(principal, plant_id, conn=conn)
        model_id = _model_id(principal, plant_id)
        obj = self._optional("model", model_id, principal, conn=conn)
        if obj is None:
            return None
        try:
            if (obj["plant_id"] != plant_id or obj["status"] != "active"
                    or set(obj["payload"]) != {"model"}
                    or digest(obj["payload"]) != obj["content_hash"]):
                raise ValueError
            bc.validate_model(obj["payload"]["model"])
        except (ValueError, TypeError, KeyError, DomainError) as exc:
            raise _integrity() from exc
        return obj

    def snapshot(self, principal: Principal, context_id: str, *, conn=None) -> dict:
        if conn is None:
            # Store.connect owns the consistent read transaction and its lifetime.
            with self.store.connect() as current:
                return self.snapshot(principal, context_id, conn=current)
        context_obj = self.get_context(principal, context_id, conn=conn)
        context = context_obj["payload"]
        model_obj = self.get_model(principal, context["plant_id"], conn=conn)
        model = model_obj["payload"]["model"] if model_obj else None
        return {
            "context": context, "context_hash": digest(context),
            "context_version": context_obj["version"], "model": model,
            "model_hash": model["model_hash"] if model else None,
            "model_content_hash": model_obj["content_hash"] if model_obj else None,
            "model_version": model_obj["version"] if model_obj else None,
            "experience_version": EXPERIENCE_VERSION, "constraint_version": CONSTRAINT_VERSION,
        }

    @staticmethod
    def _experience(context: dict) -> dict:
        query = {key: context[key] for key in ("defect", "material_id", "process_version")}
        items = []
        if (query["material_id"] == "synthetic-resin-A"
                and query["process_version"] == "synthetic-molding-v1"):
            items = [
                {"id": identifier, "text": text, "provenance": _PROVENANCE,
                 "version": EXPERIENCE_VERSION}
                for identifier, defect, text in _EXPERIENCE if defect == query["defect"]
            ]
        return {"items": items, "query": query, "context_hash": digest(context),
                "experience_version": EXPERIENCE_VERSION}

    def experience(self, principal: Principal, context_id: str) -> dict:
        return self._experience(self.get_context(principal, context_id)["payload"])

    @staticmethod
    def _device_matches(device: dict, context: dict) -> None:
        payload = device["payload"]
        expected = {key: context[key] for key in
                    ("machine_id", "plant_id", "process_version", "context_id", "current")}
        expected.update(context_hash=digest(context), synthetic=True)
        if (device["plant_id"] != context["plant_id"]
                or device["object_id"] != context["machine_id"]
                or any(not _same(payload.get(key), value) for key, value in expected.items())
                or type(payload.get("revision")) is not int or not 1 <= payload["revision"] <= 2**63 - 1
                or type(payload.get("updated_at_ms")) is not int
                or not 0 <= payload["updated_at_ms"] <= 2**63 - 1):
            raise DomainError("device_conflict", "Device is bound to a different source state", 409)

    def import_context(self, principal: Principal, context: ProcessContext, action_id: str) -> dict:
        try:
            context = ProcessContext.model_validate(context.model_dump(mode="json"))
        except (AttributeError, ValueError, TypeError) as exc:
            raise DomainError("invalid_context", "Invalid synthetic context schema", 422) from exc
        payload = context.model_dump(mode="json")
        request = {"operation": "import_context", "context_id": context.context_id,
                   "plant_id": context.plant_id, "context": payload}
        with self.store.transaction() as conn:
            self.store.require(principal, context.plant_id, conn=conn)
            existing = self._optional("context", context.context_id, principal, conn=conn)
            if existing is not None:
                self._context(existing)
                if existing["plant_id"] != context.plant_id:
                    raise DomainError("conflict", "Context identity is already bound", 409)
            device = self._optional("device", context.machine_id, principal, conn=conn)
            previous = self.store.receipt(conn, principal, action_id, request)
            if previous is not None:
                if (existing is None or device is None
                        or device["plant_id"] != context.plant_id):
                    raise _integrity()
                # A receipt is a historical response, not a command to restore
                # the original device state after a legitimate later execution.
                return previous
            if existing is not None:
                raise DomainError("conflict", "Context identity is append-only", 409)
            if device is not None:
                self._device_matches(device, payload)
            else:
                device_payload = {key: payload[key] for key in
                                  ("machine_id", "plant_id", "process_version", "context_id", "current")}
                device_payload.update(context_hash=digest(payload), revision=1, synthetic=True,
                                      updated_at_ms=self._now())
                self.store.create_object(
                    conn, kind="device", object_id=context.machine_id, principal=principal,
                    plant_id=context.plant_id, status="ready", payload=device_payload,
                )
                self.store.audit(conn, principal, kind="device", object_id=context.machine_id,
                                 action="import_device", detail={"context_hash": digest(payload)})
            obj = self.store.create_object(
                conn, kind="context", object_id=context.context_id, principal=principal,
                plant_id=context.plant_id, status="active", payload=payload,
            )
            self.store.audit(conn, principal, kind="context", object_id=context.context_id,
                             action="import_context", detail={"context_hash": digest(payload)})
            self.store.record_receipt(conn, principal, action_id, request, obj)
            return obj

    def register_model(self, principal: Principal, model: dict, plant_id: str, action_id: str) -> dict:
        model_id = _model_id(principal, plant_id)
        request = {"operation": "register_model", "model_id": model_id,
                   "plant_id": plant_id, "model": model}
        with self.store.transaction() as conn:
            self.store.require(principal, plant_id, roles=("admin", "reviewer"), conn=conn)
            bc.validate_model(model)
            current = self.get_model(principal, plant_id, conn=conn)
            previous = self.store.receipt(conn, principal, action_id, request)
            if previous is not None:
                if current is None:
                    raise _integrity()
                return previous
            kwargs = dict(kind="model", object_id=model_id, principal=principal,
                          status="active", payload={"model": model})
            if current is None:
                obj = self.store.create_object(conn, plant_id=plant_id, **kwargs)
            else:
                obj = self.store.update_object(conn, expected_version=current["version"], **kwargs)
            self.store.audit(conn, principal, kind="model", object_id=model_id,
                             action="register_model", detail={"model_hash": model["model_hash"],
                                                              "version": obj["version"]})
            self.store.record_receipt(conn, principal, action_id, request, obj)
            return obj

    def supplement_context(self, conn: sqlite3.Connection, principal: Principal, context_id: str,
                           *, expected_hash: str, fields: dict, note: str) -> dict:
        obj = self.get_context(principal, context_id, conn=conn)
        before = obj["payload"]
        if expected_hash != digest(before):
            raise DomainError("conflict", "Context source has changed", 409)
        _text(note, maximum=2000)
        bounds = {"material_viscosity": (0.3, 2.5), "ambient_c": (5.0, 45.0)}
        if type(fields) is not dict or not fields or not set(fields) <= bounds.keys():
            raise DomainError("invalid_supplement", "Only missing measurements may be supplied", 422)
        for name, value in fields.items():
            low, high = bounds[name]
            if (before[name] is not None or type(value) not in (float, int)
                    or not low <= value <= high or not math.isfinite(value)):
                raise DomainError("invalid_supplement", "Invalid missing measurement supplement", 422)
        after = {**before, **fields, "revision": before["revision"] + 1}
        try:
            after = ProcessContext.model_validate(after).model_dump(mode="json")
        except (ValueError, TypeError) as exc:
            raise DomainError("invalid_supplement", "Invalid missing measurement supplement", 422) from exc
        device = self.store.get_object("device", before["machine_id"], principal, conn=conn)
        self._device_matches(device, before)
        updated = self.store.update_object(
            conn, kind="context", object_id=context_id, principal=principal,
            expected_version=obj["version"], status="active", payload=after,
        )
        # This changes source binding only, not a setpoint, device revision, or
        # measurement timestamp. A later execution must still gate freshness.
        device_payload = {**device["payload"], "context_hash": digest(after)}
        self.store.update_object(
            conn, kind="device", object_id=device["object_id"], principal=principal,
            expected_version=device["version"], status=device["status"], payload=device_payload,
        )
        self.store.audit(conn, principal, kind="context", object_id=context_id,
                         action="supplement_context", detail={
                             "before_hash": digest(before), "after_hash": digest(after),
                             "fields": fields, "note": note,
                             "observed_at_ms": before["observed_at_ms"],
                         })
        self.store.audit(conn, principal, kind="device", object_id=device["object_id"],
                         action="supplement_source", detail={"context_hash": digest(after)})
        return updated

    def submit_recommendation(
        self, conn: sqlite3.Connection, principal: Principal, *, task_id: str,
        context_id: str, expected_snapshot: dict, candidate: dict, evidence: list[dict],
        explanation: str,
    ) -> dict:
        """Recompute trusted results and persist one immutable submission per task.

        Evidence hashes are content identities, not signatures. Provenance comes
        from matching the backend-issued evidence in the authorized stored task,
        then independently recomputing each actual tool result. This method is an
        internal Agent boundary, never an endpoint accepting arbitrary evidence.
        """
        _identifier(task_id, maximum=60)
        _text(explanation)
        task = self.store.get_object("task", task_id, principal, conn=conn)
        task_payload = task["payload"]
        snapshot = self.snapshot(principal, context_id, conn=conn)
        source = {key: snapshot[key] for key in SOURCE_KEYS}
        if (type(expected_snapshot) is not dict
                or not set(SOURCE_KEYS) <= expected_snapshot.keys()
                or not _same({key: expected_snapshot[key] for key in SOURCE_KEYS}, source)
                or not _same(task_payload.get("source_snapshot"), source)
                or task_payload.get("context_id") != context_id
                or task["plant_id"] != snapshot["context"]["plant_id"]):
            raise DomainError("source_changed", "Task source version or identity changed", 409)
        if snapshot["model"] is None:
            raise DomainError("model_missing", "An explicitly registered model is required", 409)
        if (type(candidate) is not dict or type(candidate.get("candidate_hash")) is not str
                or digest({key: value for key, value in candidate.items()
                           if key != "candidate_hash"}) != candidate["candidate_hash"]):
            raise DomainError("candidate_changed", "Frozen candidate hash does not match", 409)
        now = self._now()
        context = ProcessContext.model_validate(snapshot["context"])
        checked = check_candidate(context, candidate.get("raw_delta"), now)
        if not checked["eligible"]:
            raise DomainError("candidate_ineligible", "Current hard constraints reject the candidate", 409)
        checked.update(task_id=task_id, context_id=context_id, model_hash=snapshot["model_hash"])
        checked["candidate_hash"] = digest(checked)
        if not _same(candidate, checked) or not _same(task_payload.get("candidate"), checked):
            raise DomainError("candidate_changed", "Candidate differs from its frozen result", 409)
        prediction = bc.predict(snapshot["model"], context, now)
        if (prediction["status"] != "ok"
                or not _same(prediction["raw_delta"], checked["raw_delta"])
                or not _same(task_payload.get("prediction"), prediction)):
            raise DomainError("prediction_changed", "Candidate is not the actual eligible BC prediction", 409)
        expected_evidence = {
            "get_process_context": {"context": snapshot["context"],
                                    "issues": validate_context(context, now)},
            "retrieve_experience": self._experience(snapshot["context"]),
            "predict_bc": prediction,
            "check_constraints": {"candidate": checked},
        }
        if type(evidence) is not list or len(evidence) != len(expected_evidence):
            raise DomainError("invalid_evidence", "All required issued evidence must be supplied", 409)
        seen_ids, seen_tools = set(), set()
        for item in evidence:
            if type(item) is not dict or set(item) != {
                "evidence_id", "task_id", "tool", "context_hash", "payload", "content_hash",
            }:
                raise DomainError("invalid_evidence", "Invalid issued evidence schema", 409)
            identifier, tool = item["evidence_id"], item["tool"]
            _identifier(identifier)
            if (type(tool) is not str or tool not in expected_evidence
                    or identifier in seen_ids or tool in seen_tools
                    or item["task_id"] != task_id or item["context_hash"] != source["context_hash"]
                    or item["content_hash"] != digest({key: value for key, value in item.items()
                                                      if key != "content_hash"})
                    or not _same(item["payload"], expected_evidence[tool])):
                raise DomainError("invalid_evidence", "Evidence result, identity, or hash does not match", 409)
            seen_ids.add(identifier)
            seen_tools.add(tool)
        if not _same(task_payload.get("evidence"), evidence):
            raise DomainError("invalid_evidence", "Evidence was not issued for this stored task", 409)
        submission = {
            "task_id": task_id, "context_id": context_id, **source,
            "candidate": checked, "evidence": evidence,
            "untrusted_model_draft": explanation, "synthetic": True,
        }
        recommendation_id = "rec-" + task_id
        current = self._optional("recommendation", recommendation_id, principal, conn=conn)
        if current is not None:
            # Review/execution can append their own fields or change status. They
            # cannot alter the original submission. Replays never undo a review.
            if (current["plant_id"] != task["plant_id"]
                    or any(not _same(current["payload"].get(key), value)
                           for key, value in submission.items())):
                raise DomainError("idempotency_conflict", "Task already has a different recommendation", 409)
            if task["status"] not in {"running", "submitted"}:
                raise DomainError("task_stopped", "Task cannot submit a recommendation", 409)
            return current
        if task["status"] != "running" or task_payload.get("status") != "running":
            raise DomainError("task_stopped", "Only a running task can submit a recommendation", 409)
        obj = self.store.create_object(
            conn, kind="recommendation", object_id=recommendation_id, principal=principal,
            plant_id=task["plant_id"], status="pending_review", payload=submission,
        )
        self.store.audit(conn, principal, kind="recommendation", object_id=recommendation_id,
                         action="submit_recommendation", detail={
                             "task_id": task_id, "candidate_hash": checked["candidate_hash"],
                             "source_snapshot": source, "synthetic": True,
                         })
        return obj
