"""Run original synthetic acceptance; never connect to production systems."""

import argparse
import json
import secrets
import time
from pathlib import Path

from parameter_agent.actions import ActionService
from parameter_agent.agent import AgentEngine
from parameter_agent.bc import train_model
from parameter_agent.fixtures import demo_contexts, generate_demonstrations
from parameter_agent.provider import SyntheticProvider
from parameter_agent.service import RecommendationService
from parameter_agent.storage import Store


def run(data_root: Path, provider, *, model_mode: str, cases: list[str]):
    if not data_root.is_absolute() or not data_root.parent.is_dir() or data_root.exists():
        raise ValueError("Acceptance requires a new absolute directory with an existing parent")
    project = Path(__file__).resolve().parents[1]
    if data_root.resolve().is_relative_to(project):
        raise ValueError("Acceptance output must be outside the source repository")
    data_root.mkdir(mode=0o700)
    started = time.monotonic()
    current_ms = [int(time.time() * 1000)]
    report = {"status": "failed", "model_mode": model_mode, "synthetic_data": True,
              "cases": [], "not_field_validation": True}
    try:
        store = Store(data_root / "data")
        principal = store.create_user("synthetic-admin", secrets.token_urlsafe(32), "admin",
                                      ["synthetic-plant"])
        service = RecommendationService(store, clock=lambda: current_ms[0])
        actions = ActionService(service, clock=lambda: current_ms[0])
        training = train_model(generate_demonstrations())
        model = service.register_model(principal, training["model"], "synthetic-plant", "train-1")
        report["bc_metrics"] = training["metrics"]
        report["model_hash"] = model["payload"]["model"]["model_hash"]
        contexts = demo_contexts(current_ms[0])
        for name, context in contexts.items():
            service.import_context(principal, context, "import-" + name)
        for name in cases:
            engine = AgentEngine(service, provider, clock=lambda: current_ms[0])
            task = engine.create(principal, contexts[name].context_id, action_id="create-" + name)
            task = engine.run(principal, task["object_id"])
            if name == "missing":
                assert task["status"] == "awaiting_input", task["payload"].get("error")
                before_calls = task["payload"]["calls"]
                required = task["payload"]["awaiting_fields"]
                values = {"material_viscosity": 1.0, "ambient_c": 25.0}
                task = engine.resume(principal, task["object_id"], action_id="supplement-" + name,
                                     fields={key: values[key] for key in required},
                                     note="Explicit original synthetic context supplement")
                assert task["payload"]["calls"] == before_calls
                task = engine.run(principal, task["object_id"])
            row = {"case": name, "status": task["status"], "calls": task["payload"]["calls"],
                   "tools": task["payload"]["tools"], "tokens": task["payload"]["tokens"],
                   "error": task["payload"]["error"]}
            report["cases"].append(row)
            if name in {"normal", "ood"}:
                assert task["status"] == "no_recommendation", row
                row["checks_passed"] = True
                continue
            assert task["status"] == "submitted", row
            terminal = engine.run(principal, task["object_id"])
            assert terminal["payload"]["calls"] == task["payload"]["calls"]
            recommendation = store.get_object("recommendation", task["payload"]["recommendation_id"],
                                              principal)
            assert recommendation["status"] == "pending_review"
            frozen_candidate = recommendation["payload"]["candidate"]
            recommendation = actions.review(
                principal, recommendation["object_id"], action_id="review-" + name,
                recommendation_hash=recommendation["content_hash"], decision="approved",
                note="Explicit original synthetic approval, not a production signature",
            )
            execution = actions.request_execution(
                principal, recommendation["object_id"], action_id="request-" + name,
                recommendation_hash=recommendation["content_hash"], confirmed=True,
            )
            assert execution["status"] == "pending"
            execution = actions.apply_synthetic(principal, execution["object_id"], action_id="apply-" + name)
            assert execution["status"] == "applied"
            execution = actions.readback(principal, execution["object_id"], action_id="read-" + name)
            assert execution["status"] == "executed"
            # Explicit injected synthetic clock: do not pretend to wait for physical samples.
            current_ms[0] += 61000
            observed = actions.observe(principal, recommendation["object_id"], action_id="observe-" + name)
            assert observed["status"] == "observed"
            assert observed["payload"]["candidate"] == frozen_candidate
            row.update(lifecycle="observed", checks_passed=True, synthetic_clock_advanced=True,
                       causal_effect_proven=False)
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "code": getattr(exc, "code", "acceptance_failed")}
        raise
    finally:
        close = getattr(provider, "close", None)
        cleanup_error = None
        if close is not None:
            try:
                close()
            except Exception as exc:
                cleanup_error = exc
                report["status"] = "failed"
                report["cleanup_error"] = getattr(exc, "code", "cleanup_unknown")
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        with (data_root / "acceptance.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps({k: v for k, v in report.items() if k != "bc_metrics"}, ensure_ascii=False))
        if cleanup_error is not None:
            raise RuntimeError("Acceptance runtime cleanup could not be confirmed") from None
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test-provider", action="store_true")
    mode.add_argument("--runtime-config", type=Path)
    parser.add_argument("--case", action="append", choices=["short_shot", "flash", "warpage", "normal", "missing", "ood"])
    args = parser.parse_args()
    if args.test_provider:
        provider = SyntheticProvider()
        model_mode = "TEST ONLY"
    else:
        from parameter_agent.managed_provider import ManagedLocalProvider
        provider = ManagedLocalProvider(args.runtime_config)
        model_mode = "real-local"
    cases = args.case or (["short_shot", "flash", "warpage", "normal", "missing", "ood"]
                         if args.test_provider else ["short_shot"])
    run(args.data_root, provider, model_mode=model_mode, cases=cases)


if __name__ == "__main__":
    main()
