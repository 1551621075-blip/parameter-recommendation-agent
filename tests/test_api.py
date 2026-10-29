"""HTTP acceptance uses TestClient, real SQLite, and explicit TEST/mocked boundaries only."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from parameter_agent import api
from parameter_agent.domain import ProcessContext
from parameter_agent.fixtures import demo_contexts

NOW = 1_780_000_000_000
BASE = "http://127.0.0.1:8005"
TOKEN = "".join(("api-test-owner-token-", "a" * 24))
OTHER_TOKEN = "".join(("api-test-other-token-", "b" * 24))


class NoInferenceProvider:
    fingerprint = "TEST-boundary-no-inference"
    is_test_provider = True

    def decide(self, context):
        raise AssertionError("This HTTP test must not invoke inference")


@pytest.fixture
def boundary(tmp_path):
    application = api.create_app(tmp_path.resolve() / "data", NoInferenceProvider(), clock=lambda: NOW)
    context = demo_contexts(NOW)["short_shot"]
    store = application.state.store
    owner = store.create_user("api-owner", TOKEN, "admin", [context.plant_id])
    other = store.create_user("api-other", OTHER_TOKEN, "admin", [context.plant_id])
    with TestClient(application, base_url=BASE) as client:
        yield application, client, owner, other, context


def auth(token=TOKEN, **extra):
    return {"Authorization": "Bearer " + token, **extra}


def import_context(client, context, action_id="import-one"):
    result = client.post("/api/contexts", headers=auth(Origin=BASE),
                         json={"action_id": action_id, "context": context.model_dump(mode="json")})
    assert result.status_code == 200, result.text
    return result.json()


def create_task(client, context):
    import_context(client, context)
    result = client.post("/api/tasks", headers=auth(),
                         json={"action_id": "create-task", "context_id": context.context_id})
    assert result.status_code == 200, result.text
    return result.json()


def test_health_is_public_noninference_and_app_seeds_nothing(tmp_path):
    application = api.create_app(tmp_path.resolve() / "empty", NoInferenceProvider())
    with TestClient(application, base_url=BASE) as client:
        response = client.get("/health")
        assert response.json() == {"status": "ok", "provider": "TEST-boundary-no-inference",
                                   "synthetic_only": True}
        assert response.status_code == 200
        with application.state.store.connect() as connection:
            for table in ("users", "objects", "calls", "audit"):
                assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        assert client.get("/api/me").status_code == 401
    assert application.openapi_url is None
    assert application.docs_url is None
    assert application.redoc_url is None


def test_static_assets_are_public_local_and_have_security_headers(boundary):
    _, client, *_ = boundary
    for path in ("/", "/static/app.js", "/static/style.css", "/health"):
        response = client.get(path)
        assert response.status_code == 200, (path, response.text)
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
        assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize("path", ["/openapi.json", "/docs", "/redoc", "/static/other.js",
                                  "/api/models/upload", "/api/path", "/api/users"])
def test_no_docs_upload_or_administration_endpoints(boundary, path):
    _, client, *_ = boundary
    assert client.get(path, headers=auth()).status_code == 404
    if path.startswith("/api/"):
        assert client.post(path, headers=auth(), json={}).status_code == 404


@pytest.mark.parametrize("host", ["localhost:8005", "evil.example:8005", "0.0.0.0:8005",
                                  "192.168.1.5:8005", "127.0.0.1.evil:8005", "2130706433:8005",
                                  "127.1:8005", "127.0.0.1:0", "127.0.0.1:65536",
                                  "127.0.0.1:", "127.0.0.1:8005@evil", "[::1]junk:8005",
                                  "127.0.0.1:+8005", "[::1%25lo0]:8005", "[::ffff:127.0.0.1]:8005"])
def test_numeric_loopback_host_is_mandatory(boundary, host):
    _, client, *_ = boundary
    response = client.get("/health", headers={"Host": host,
                          "X-Forwarded-Host": "127.0.0.1:8005"})
    assert response.status_code == 400
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("host", ["127.0.0.1:8005", "127.0.0.2", "[::1]:8005", "[0:0:0:0:0:0:0:1]"])
def test_numeric_loopback_hosts_work_without_proxy_trust(boundary, host):
    _, client, *_ = boundary
    assert client.get("/health", headers={"Host": host, "X-Forwarded-Host": "evil.example"}).status_code == 200


def test_duplicate_sensitive_headers_are_rejected(boundary):
    _, client, *_ = boundary
    assert client.get("/health", headers=[("Host", "127.0.0.1"), ("Host", "127.0.0.1")]).status_code == 400
    headers = [("Authorization", "Bearer " + TOKEN), ("Authorization", "Bearer " + TOKEN)]
    assert client.get("/api/me", headers=headers).status_code == 401
    headers = [("Authorization", "Bearer " + TOKEN), ("Content-Length", "2"),
               ("Content-Length", "2"), ("Content-Type", "application/json")]
    assert client.post("/api/tasks/task-one/run", headers=headers, content=b"{}").status_code == 400


@pytest.mark.parametrize("extra", [
    {"Origin": "http://evil.example"}, {"Origin": "null"}, {"Origin": ""},
    {"Origin": "http://localhost:8005"}, {"Origin": "http://127.0.0.1:8006"},
    {"Origin": "https://127.0.0.1:8005"}, {"Origin": BASE + "/"},
    {"Origin": BASE + "?x=1"}, {"Sec-Fetch-Site": "cross-site"},
    {"Sec-Fetch-Site": "same-site"}, {"Referer": "http://evil.example/form"},
])
def test_mutations_reject_cross_origin_even_with_bearer(boundary, extra):
    _, client, *_ = boundary
    response = client.post("/api/tasks/no-task/run", headers=auth(**extra), json={})
    assert response.status_code == 403


@pytest.mark.parametrize("extra", [{}, {"Origin": BASE}, {"Referer": BASE + "/workbench"},
                                    {"Sec-Fetch-Site": "same-origin", "Origin": BASE}])
def test_native_clients_or_same_origin_can_mutate(boundary, extra):
    _, client, _, _, context = boundary
    response = client.post("/api/contexts", headers=auth(**extra),
                           json={"action_id": "context-one", "context": context.model_dump(mode="json")})
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("authorization", ["", "Basic abc", "Bearer nope", "Bearer  " + TOKEN,
                                            "Bearer " + TOKEN + " ", "Bearer " + "a" * 201])
def test_bearer_authentication_required_without_echo(boundary, authorization):
    _, client, *_ = boundary
    response = client.get("/api/me", headers={"Authorization": authorization})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert TOKEN not in response.text


def test_identity_and_plant_scope_are_server_owned(boundary):
    application, client, owner, _, context = boundary
    first = import_context(client, context)
    assert first["owner"] == owner.user_id
    assert client.get("/api/me", headers=auth()).json() == {
        "user_id": owner.user_id, "role": "admin", "plants": list(owner.plants), "acl_version": 1}
    assert len(client.get("/api/contexts", headers=auth()).json()["items"]) == 1
    assert client.get("/api/contexts", headers=auth(OTHER_TOKEN)).json() == {"items": []}
    assert client.get("/api/audit", headers=auth(OTHER_TOKEN)).json() == {"items": []}
    events = client.get("/api/audit", headers=auth()).json()["items"]
    assert events and TOKEN not in json.dumps(events)
    for query in ("token=" + TOKEN, "role=admin", "owner=api-other"):
        response = client.get("/api/me?" + query, headers=auth())
        assert response.status_code == 400
        assert TOKEN not in response.text
    application.state.store.set_user(owner.user_id, plants=[])
    assert client.get("/api/contexts", headers=auth()).json() == {"items": []}
    denied = client.post("/api/contexts", headers=auth(),
                         json={"action_id": "import-one", "context": context.model_dump(mode="json")})
    assert denied.status_code in {403, 404}
    application.state.store.set_user(owner.user_id, active=False)
    assert client.get("/api/me", headers=auth()).status_code == 401


@pytest.mark.parametrize("body", [b'[]', b'null', b'"text"', b'', b'{"x":1,"x":2}',
                                  b'{"x":{"same":1,"same":2}}', b'{"x":NaN}',
                                  b'{"x":Infinity}', b'{"x":-Infinity}', b'{"x":1e309}',
                                  b'{"x":9223372036854775808}', b'{"x":"\\ud800"}',
                                  b'{"x":"\xff"}', b'\xef\xbb\xbf{}',
                                  b'{"x":' + b'[' * 40 + b'0' + b']' * 40 + b'}'])
def test_json_is_unique_finite_bounded_utf8_object(boundary, body):
    _, client, *_ = boundary
    response = client.post("/api/tasks/task-one/run", headers=auth(**{"Content-Type": "application/json"}),
                           content=body)
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "invalid_json"


@pytest.mark.parametrize("content_type", ["text/plain", "application/json; charset=utf-16",
                                          "application/json; arbitrary=1", "application/x-www-form-urlencoded"])
def test_only_uncompressed_utf8_json_is_accepted(boundary, content_type):
    _, client, *_ = boundary
    assert client.post("/api/tasks/task-one/run", content=b"{}",
                       headers=auth(**{"Content-Type": content_type})).status_code == 415
    assert client.post("/api/tasks/task-one/run", content=b"{}", headers=auth(**{
        "Content-Type": "application/json", "Content-Encoding": "gzip"})).status_code == 415


def test_schema_strictness_and_idempotent_import(boundary):
    _, client, _, _, context = boundary
    obj = import_context(client, context)
    assert import_context(client, context) == obj
    base = {"action_id": "import-one", "context": context.model_dump(mode="json")}
    modified = json.loads(json.dumps(base))
    modified["context"]["severity"] = 0.75
    assert client.post("/api/contexts", headers=auth(), json=modified).status_code == 409
    invalid_bodies = [dict(base, owner="someone"), dict(base, role="admin"), dict(base, token=TOKEN)]
    for field, value in (("temperature", 204), ("temperature", 204.0), ("temperature", True)):
        invalid = json.loads(json.dumps(base))
        invalid["context"]["current"][field] = value
        invalid_bodies.append(invalid)
    for field in ("revision", "sample_count"):
        invalid = json.loads(json.dumps(base))
        invalid["context"][field] = True
        invalid_bodies.append(invalid)
    for invalid in invalid_bodies:
        response = client.post("/api/contexts", headers=auth(), json=invalid)
        assert response.status_code == 422
        assert TOKEN not in response.text


def test_declared_and_actual_body_limits_and_framing(boundary):
    _, client, *_ = boundary
    route = "/api/tasks/task-one/run"
    headers = auth(**{"Content-Type": "application/json"})
    assert client.post(route, headers=headers, content=b" " * (api.MAX_BODY + 1)).status_code == 413
    oversized = {**headers, "Content-Length": str(api.MAX_BODY + 1)}
    assert client.post(route, headers=oversized, content=b"{}").status_code == 413
    assert client.post(route, headers=headers, content=iter([b" " * api.MAX_BODY, b" "])).status_code == 413
    for length in ("-1", "+2", "2,2", "", "1", "3"):
        assert client.post(route, headers={**headers, "Content-Length": length}, content=b"{}").status_code == 400
    assert client.post(route, headers={**headers, "Content-Length": "2", "Transfer-Encoding": "chunked"},
                       content=b"{}").status_code == 400
    assert client.request("GET", "/health", headers=headers, content=b"{}").status_code == 400
    assert client.request("TRACE", "/health").status_code == 405


def test_body_deadline_includes_waiting_for_stream(boundary, monkeypatch):
    application, _, *_ = boundary
    monkeypatch.setattr(api, "MAX_BODY_SECONDS", 0.01)

    class SlowReceive:
        async def __call__(self, scope, receive, send):
            async def slow():
                await asyncio.sleep(0.05)
                return await receive()
            await application(scope, slow if scope["type"] == "http" else receive, send)

    with TestClient(SlowReceive(), base_url=BASE) as client:
        response = client.post("/api/tasks/task-one/run", headers=auth(), json={})
    assert response.status_code == 408
    assert response.json()["error"]["code"] == "body_timeout"


def test_tasks_create_cancel_list_and_owner_isolation(boundary):
    _, client, _, _, context = boundary
    task = create_task(client, context)
    route = "/api/tasks/" + task["object_id"]
    assert task["status"] == "created"
    assert task["payload"]["calls"] == 0
    assert client.get(route, headers=auth()).json()["object_id"] == task["object_id"]
    assert client.get("/api/tasks", headers=auth()).json()["items"][0]["object_id"] == task["object_id"]
    assert client.get(route, headers=auth(OTHER_TOKEN)).status_code == 404
    assert client.post(route + "/cancel", headers=auth(OTHER_TOKEN), json={"action_id": "wrong"}).status_code == 404
    assert client.post(route + "/run", headers=auth(), json={"action_id": "not-empty"}).status_code == 422
    cancelled = client.post(route + "/cancel", headers=auth(), json={"action_id": "cancel-one"})
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert client.get("/api/recommendations", headers=auth()).json() == {"items": []}
    assert client.get("/api/executions", headers=auth()).json() == {"items": []}


def test_acl_change_after_authentication_is_rechecked_inside_route(boundary, monkeypatch):
    application, client, owner, *_ = boundary
    original = application.state.store.authenticate

    def changed(token):
        principal = original(token)
        application.state.store.set_user(owner.user_id, plants=[])
        return principal

    monkeypatch.setattr(application.state.store, "authenticate", changed)
    assert client.get("/api/me", headers=auth()).status_code == 403
    assert client.get("/api/contexts", headers=auth()).status_code == 403


def test_explicit_test_provider_runs_missing_input_without_inventing_model_or_approval(boundary):
    from parameter_agent.provider import SyntheticProvider

    application, client, _, _, _ = boundary
    test_provider = SyntheticProvider()
    application.state.engine.provider = test_provider
    context = demo_contexts(NOW)["missing"]
    task = create_task(client, context)
    route = "/api/tasks/" + task["object_id"]
    result = client.post(route + "/run", headers=auth(), json={})
    assert result.status_code == 200, result.text
    payload = result.json()["payload"]
    assert payload["status"] == "awaiting_input", (payload["error"], payload["calls"], payload["evidence"])
    assert payload["calls"] > 0
    assert set(payload["awaiting_fields"]) == {"material_viscosity", "ambient_c"}
    assert client.get("/api/models", headers=auth()).json() == {"items": []}
    assert client.get("/api/recommendations", headers=auth()).json() == {"items": []}
    assert client.get("/api/executions", headers=auth()).json() == {"items": []}
    supplement = client.post(route + "/supplement", headers=auth(), json={
        "action_id": "supplement-measured", "fields": {"material_viscosity": 1.05, "ambient_c": 23.0},
        "note": "Explicit synthetic measured values"})
    assert supplement.status_code == 200, supplement.text
    assert supplement.json()["payload"]["calls"] == payload["calls"]
    assert supplement.json()["status"] != "running"
    with application.state.store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == payload["calls"]


def test_run_does_not_block_cancel_or_return_background_success(boundary, monkeypatch):
    application, client, _, _, context = boundary
    task = create_task(client, context)
    route = "/api/tasks/" + task["object_id"]
    entered, release = threading.Event(), threading.Event()

    def blocking_run(principal, task_id):
        entered.set()
        assert release.wait(5)
        return application.state.engine.get(principal, task_id)

    monkeypatch.setattr(application.state.engine, "run", blocking_run)
    with ThreadPoolExecutor(max_workers=1) as workers:
        future = workers.submit(client.post, route + "/run", headers=auth(), json={})
        try:
            assert entered.wait(5)
            assert not future.done()
            response = client.post(route + "/cancel", headers=auth(), json={"action_id": "cancel-running"})
            assert response.status_code == 200, response.text
            assert response.json()["status"] == "cancelled"
        finally:
            release.set()
        assert future.result(timeout=5).json()["status"] == "cancelled"


@pytest.mark.parametrize(("path", "method", "body", "expected"), [
    ("/api/tasks/task-one/supplement", "resume", {"action_id": "a1", "fields": {"ambient_c": 22.0}, "note": "Measured"},
     {"action_id": "a1", "fields": {"ambient_c": 22.0}, "note": "Measured"}),
    ("/api/tasks/task-one/reconcile", "reconcile", {"action_id": "a1", "call_id": "call-one", "no_result": True, "note": "Confirmed no result"},
     {"action_id": "a1", "call_id": "call-one", "no_result": True, "tokens": None, "note": "Confirmed no result"}),
    ("/api/recommendations/rec-one/review", "review", {"action_id": "a1", "recommendation_hash": "a" * 64, "decision": "approved", "note": "Reviewed synthetic parameters"},
     {"action_id": "a1", "recommendation_hash": "a" * 64, "decision": "approved", "note": "Reviewed synthetic parameters"}),
    ("/api/recommendations/rec-one/request-execution", "request_execution", {"action_id": "a1", "recommendation_hash": "a" * 64, "confirmed": True},
     {"action_id": "a1", "recommendation_hash": "a" * 64, "confirmed": True}),
    ("/api/recommendations/rec-one/observe", "observe", {"action_id": "a1"}, {"action_id": "a1"}),
    ("/api/executions/exec-one/apply", "apply_synthetic", {"action_id": "a1"}, {"action_id": "a1"}),
    ("/api/executions/exec-one/readback", "readback", {"action_id": "a1"}, {"action_id": "a1"}),
])
def test_routes_use_exact_service_interfaces(boundary, monkeypatch, path, method, body, expected):
    application, client, owner, *_ = boundary
    target = application.state.engine if path.startswith("/api/tasks/") else application.state.actions
    calls = []

    def delegated(principal, object_id, **kwargs):
        calls.append((principal, object_id, kwargs))
        return {"delegated": True}

    monkeypatch.setattr(target, method, delegated)
    response = client.post(path, headers=auth(), json=body)
    assert response.status_code == 200, response.text
    assert calls == [(owner, path.split("/")[3], expected)]


@pytest.mark.parametrize(("path", "body"), [
    ("/api/executions/e/apply", {"action_id": "a", "fault": "after_apply"}),
    ("/api/recommendations/r/observe", {"action_id": "a", "seed": 1}),
    ("/api/recommendations/r/observe", {"action_id": "a", "sample_count": 1}),
    ("/api/recommendations/r/request-execution", {"action_id": "a", "recommendation_hash": "a" * 64, "confirmed": 1}),
    ("/api/tasks/t/reconcile", {"action_id": "a", "call_id": "c", "no_result": 1, "note": "x"}),
    ("/api/tasks/t/reconcile", {"action_id": "a", "call_id": "c", "no_result": True, "tokens": True, "note": "x"}),
    ("/api/tasks/t/supplement", {"action_id": "a", "fields": {"ambient_c": True}, "note": "x"}),
])
def test_no_fault_seed_or_coerced_confirmations(boundary, path, body):
    _, client, *_ = boundary
    assert client.post(path, headers=auth(), json=body).status_code == 422


def test_foreign_plant_import_rejected(boundary):
    _, client, _, _, context = boundary
    data = context.model_dump(mode="json")
    data.update(context_id="foreign-context", machine_id="foreign-machine", plant_id="foreign-plant")
    foreign = ProcessContext.model_validate(data)
    response = client.post("/api/contexts", headers=auth(),
                           json={"action_id": "foreign-import", "context": foreign.model_dump(mode="json")})
    assert response.status_code == 403


def test_model_listing_projects_metadata_without_weights(boundary, monkeypatch):
    application, client, owner, _, context = boundary
    metadata = {"kind": "model", "object_id": "active-model-" + context.plant_id,
                "owner": owner.user_id, "plant_id": context.plant_id, "status": "active",
                "version": 1, "content_hash": "c" * 64}
    full = {**metadata, "payload": {"model": {
        "model_hash": "d" * 64, "weights": {"private": "weights-must-not-leak"},
        "metrics": {"synthetic": True, "training": {"final_loss": 0.25}},
    }}}
    original = application.state.store.list_objects

    def listing(kind, principal):
        if kind == "model":
            assert principal == owner
            return [full]
        return original(kind, principal)

    monkeypatch.setattr(application.state.store, "list_objects", listing)
    response = client.get("/api/models", headers=auth())
    assert response.status_code == 200
    projected = response.json()["items"][0]
    assert {key: projected[key] for key in metadata} == metadata
    assert projected["payload"]["model_hash"] == "d" * 64
    assert projected["payload"]["metrics"]["test"] is None
    assert "weights-must-not-leak" not in response.text
    assert '"weights"' not in response.text
    assert client.post("/api/models", headers=auth(), json={"model": full}).status_code == 405
    # A scoped evaluation with the wrong model identity is corruption, never a fallback.
    from parameter_agent.domain import digest

    metrics_id = "metrics-" + digest({"owner": owner.user_id, "plant_id": context.plant_id,
                                     "model_hash": "d" * 64})
    with application.state.store.transaction() as connection:
        application.state.store.create_object(
            connection, kind="model_metrics", object_id=metrics_id, principal=owner,
            plant_id=context.plant_id, status="evaluated", payload={"model_hash": "e" * 64,
                                                                   "metrics": {"synthetic": True}})
    corrupt = client.get("/api/models", headers=auth())
    assert corrupt.status_code == 409
    assert corrupt.json()["error"]["code"] == "integrity"


def test_recommendation_review_authorization_and_execution_are_separate(boundary):
    application, client, owner, _, context = boundary
    store = application.state.store
    with store.transaction() as connection:
        recommendation = store.create_object(
            connection, kind="recommendation", object_id="rec-separation", principal=owner,
            plant_id=context.plant_id, status="pending_review", payload={"synthetic": True,
            "candidate": {"target": context.current.model_dump(mode="json")},
            "untrusted_model_draft": "Synthetic test fixture, not a trained recommendation"})
    route = "/api/recommendations/" + recommendation["object_id"]
    assert client.get(route, headers=auth(OTHER_TOKEN)).status_code == 404
    request = {"action_id": "review-fixture", "recommendation_hash": recommendation["content_hash"],
               "decision": "approved", "note": "Explicit synthetic fixture approval"}
    assert client.post(route + "/review", headers=auth(OTHER_TOKEN), json=request).status_code == 404
    store.set_user(owner.user_id, role="operator")
    assert client.post(route + "/review", headers=auth(), json=request).status_code == 403
    store.set_user(owner.user_id, role="reviewer")
    approved = client.post(route + "/review", headers=auth(), json=request)
    assert approved.status_code == 200, approved.text
    assert approved.json()["status"] == "approved"
    assert client.get("/api/executions", headers=auth()).json() == {"items": []}
    # API fixtures cannot manufacture the missing source/model/evidence required for execution.
    execute = client.post(route + "/request-execution", headers=auth(), json={
        "action_id": "execute-fixture", "recommendation_hash": approved.json()["content_hash"],
        "confirmed": True})
    assert execute.status_code in {404, 409, 422}
    assert client.get("/api/executions", headers=auth()).json() == {"items": []}


def test_internal_failures_are_sanitized(boundary, monkeypatch):
    application, client, *_ = boundary

    def fail(*args, **kwargs):
        raise RuntimeError("sensitive-data-" + TOKEN)

    monkeypatch.setattr(application.state.store, "audit_events", fail)
    response = client.get("/api/audit", headers=auth())
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert TOKEN not in response.text
    assert "sensitive-data" not in response.text
