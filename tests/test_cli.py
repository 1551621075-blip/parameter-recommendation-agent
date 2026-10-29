"""CLI acceptance never starts a server or real language model and uses private test databases."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from parameter_agent import bc, cli
from parameter_agent.domain import ProcessContext
from parameter_agent.fixtures import generate_demonstrations
from parameter_agent.storage import Store

RUNNER = CliRunner()


@pytest.fixture
def initialized(tmp_path):
    # Explicit test setup isolates train/serve acceptance from fixture-import integration.
    path = tmp_path.resolve() / "local-data"
    store = Store(path)
    token = "".join(("cli-test-admin-token-", "a" * 24))
    principal = store.create_user("cli-admin", token, "admin", ["synthetic-plant"])
    return path, {"token": token}, store, principal


def test_init_imports_only_contexts_and_hashes_token_once(tmp_path, monkeypatch):
    training = Mock(side_effect=AssertionError("init must not train"))
    monkeypatch.setattr(cli.bc, "train_model", training)
    path = tmp_path.resolve() / "new-local-data"
    result = RUNNER.invoke(cli.app, ["init", str(path)])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    token = report["token"]
    assert result.output.count(token) == 1
    assert len(token) >= 32
    assert report["role"] == "admin"
    assert report["plants"] == ["synthetic-plant"]
    assert report["contexts_imported"] == 6
    assert path.stat().st_mode & 0o777 == 0o700
    store = Store(path)
    principal = store.authenticate(token)
    assert len(store.list_objects("context", principal)) == 6
    assert len(store.list_objects("device", principal)) == 6
    for kind in ("model", "task", "recommendation", "execution", "device_receipt"):
        assert store.list_objects(kind, principal) == []
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 0
        user = connection.execute("SELECT * FROM users").fetchone()
        assert token not in str(dict(user))
    for file in path.rglob("*"):
        if file.is_file():
            assert token.encode() not in file.read_bytes()
    training.assert_not_called()
    again = RUNNER.invoke(cli.app, ["init", str(path)])
    assert again.exit_code != 0
    assert token not in again.output
    assert store.authenticate(token) == principal


def test_init_never_overwrites_or_creates_missing_parent(tmp_path):
    existing = tmp_path.resolve() / "existing"
    existing.mkdir()
    assert RUNNER.invoke(cli.app, ["init", str(existing)]).exit_code != 0
    missing = tmp_path.resolve() / "missing" / "child"
    assert RUNNER.invoke(cli.app, ["init", str(missing)]).exit_code != 0
    assert not missing.parent.exists()
    regular = tmp_path.resolve() / "file"
    regular.write_text("do not overwrite")
    assert RUNNER.invoke(cli.app, ["init", str(regular)]).exit_code != 0
    assert regular.read_text() == "do not overwrite"


def test_cli_rejects_symlink_and_parent_traversal(tmp_path):
    root = tmp_path.resolve()
    real = root / "real"
    real.mkdir()
    link = root / "link"
    link.symlink_to(real, target_is_directory=True)
    response = RUNNER.invoke(cli.app, ["init", str(link / "new")])
    assert response.exit_code != 0
    assert "unsafe_path" in response.output
    assert not (real / "new").exists()
    response = RUNNER.invoke(cli.app, ["init", str(real / ".." / "new")])
    assert response.exit_code != 0
    assert "unsafe_path" in response.output
    assert not (root / "new").exists()


@pytest.mark.parametrize("case", cli.FIXTURE_CASES)
def test_fixture_fixed_cases_output_strict_original_json(case):
    result = RUNNER.invoke(cli.app, ["fixture", "--case", case])
    assert result.exit_code == 0, result.output
    context = ProcessContext.model_validate(json.loads(result.stdout))
    assert context.synthetic is True
    assert context.context_id.startswith("synthetic-")
    assert all(type(value) is str for value in context.current.model_dump().values())
    if case == "missing":
        assert context.material_viscosity is None
        assert context.ambient_c is None


def test_fixture_rejects_unknown_case_and_has_no_path_option():
    assert RUNNER.invoke(cli.app, ["fixture", "--case", "../../secret"]).exit_code != 0
    assert RUNNER.invoke(cli.app, ["fixture", "--path", "/private/model"]).exit_code != 0


def test_train_does_actual_bc_and_registers_safe_local_artifacts(initialized, monkeypatch):
    path, report, store, principal = initialized
    calls = []
    original = bc.train_model

    def actual_train(records):
        calls.append(len(records))
        return original(records)

    monkeypatch.setattr(cli, "generate_demonstrations", lambda: generate_demonstrations(groups=10, per_group=4))
    monkeypatch.setattr(cli.bc, "train_model", actual_train)
    result = RUNNER.invoke(cli.app, ["train", str(path)], input=report["token"] + "\n")
    assert result.exit_code == 0, result.output
    assert calls == [40]
    assert report["token"] not in result.output
    output = json.loads(result.stdout[result.stdout.index("{"):])
    assert output["metrics"]["synthetic"] is True
    training = output["metrics"]["training"]
    assert training["final_loss"] < training["initial_loss"]
    assert training["test_used_for_tuning"] is False
    assert output["metrics"]["test"]
    model = store.get_artifact(output["model_artifact"])
    metrics = store.get_artifact(output["metrics_artifact"])
    bc.validate_model(model)
    assert metrics == {"model_hash": model["model_hash"], "metrics": output["metrics"]}
    registered = store.get_object("model", output["model_id"], principal)
    assert registered["payload"]["model"] == model
    assert registered["plant_id"] == "synthetic-plant"
    assert registered["version"] == 1
    evaluations = store.list_objects("model_metrics", principal)
    assert len(evaluations) == 1
    assert evaluations[0]["status"] == "evaluated"
    assert evaluations[0]["payload"] == metrics
    from fastapi.testclient import TestClient

    from parameter_agent.api import create_app

    provider = Mock(fingerprint="TEST-no-inference")
    application = create_app(path, provider)
    with TestClient(application, base_url="http://127.0.0.1:8005") as client:
        response = client.get("/api/models", headers={"Authorization": "Bearer " + report["token"]})
        assert response.status_code == 200, response.text
        item = response.json()["items"][0]
        assert item["payload"]["metrics"] == output["metrics"]
        assert '"weights"' not in response.text
    provider.decide.assert_not_called()
    for artifact in (output["model_artifact"], output["metrics_artifact"]):
        file = path / "artifacts" / (artifact + ".json")
        assert file.is_file() and not file.is_symlink()
        assert file.stat().st_nlink == 1
        assert file.stat().st_mode & 0o777 == 0o600
    for kind in ("task", "recommendation", "execution"):
        assert store.list_objects(kind, principal) == []


def test_train_denies_bad_token_operator_and_revoked_acl_before_training(initialized, monkeypatch):
    path, report, store, principal = initialized
    trainer = Mock(side_effect=AssertionError("Unauthorized training must not start"))
    monkeypatch.setattr(cli.bc, "train_model", trainer)
    wrong = RUNNER.invoke(cli.app, ["train", str(path)], input="wrong-not-a-token\n")
    assert wrong.exit_code != 0
    assert "wrong-not-a-token" not in wrong.output
    store.set_user(principal.user_id, role="operator")
    operator = RUNNER.invoke(cli.app, ["train", str(path)], input=report["token"] + "\n")
    assert operator.exit_code != 0
    assert report["token"] not in operator.output
    store.set_user(principal.user_id, role="admin", plants=[])
    revoked = RUNNER.invoke(cli.app, ["train", str(path)], input=report["token"] + "\n")
    assert revoked.exit_code != 0
    trainer.assert_not_called()


def test_train_failure_does_not_claim_success_or_register_model(initialized, monkeypatch):
    path, report, store, principal = initialized
    monkeypatch.setattr(cli, "generate_demonstrations", lambda: [])
    monkeypatch.setattr(cli.bc, "train_model", Mock(side_effect=RuntimeError("secret-internal-text")))
    result = RUNNER.invoke(cli.app, ["train", str(path)], input=report["token"] + "\n")
    assert result.exit_code != 0
    assert "secret-internal-text" not in result.output
    assert report["token"] not in result.output
    assert store.list_objects("model", principal) == []


def test_train_does_not_accept_token_arguments_or_environment(initialized, monkeypatch):
    path, report, *_ = initialized
    monkeypatch.setenv("PARAMETER_TOKEN", report["token"])
    assert RUNNER.invoke(cli.app, ["train", str(path), "--token", report["token"]]).exit_code != 0
    result = RUNNER.invoke(cli.app, ["train", str(path)], input="\n")
    assert result.exit_code != 0


@pytest.mark.parametrize("command", ["train", "serve"])
def test_existing_store_required_no_implicit_init(tmp_path, command):
    path = tmp_path.resolve() / "missing-data"
    args = [command, str(path)] + (["--test-mode"] if command == "serve" else [])
    result = RUNNER.invoke(cli.app, args)
    assert result.exit_code != 0
    assert not path.exists()


def test_training_rechecks_acl_after_compute_before_publish(initialized, monkeypatch):
    path, report, store, principal = initialized
    monkeypatch.setattr(cli, "generate_demonstrations", lambda: [])

    def revoke(_records):
        store.set_user(principal.user_id, active=False)
        return {"model": {}, "metrics": {}}

    monkeypatch.setattr(cli.bc, "train_model", revoke)
    monkeypatch.setattr(cli.bc, "validate_model", lambda _model: None)
    result = RUNNER.invoke(cli.app, ["train", str(path)], input=report["token"] + "\n")
    assert result.exit_code != 0
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM objects WHERE kind='model'").fetchone()[0] == 0


def test_serve_requires_exactly_one_explicit_provider(initialized):
    path, *_ = initialized
    for options in ([], ["--test-mode", "--model-url", "http://127.0.0.1:9999"],
                    ["--test-mode", "--runtime-config", "/not-read.json"],
                    ["--runtime-config", "/not-read.json", "--model-url", "http://127.0.0.1:9999"]):
        result = RUNNER.invoke(cli.app, ["serve", str(path), *options])
        assert result.exit_code != 0
        assert "provider_choice" in result.output


@pytest.mark.parametrize("options", [["--port", "0"], ["--port", "65536"], ["--host", "0.0.0.0"],
                                     ["--host", "localhost"], ["--host", "::1"]])
def test_serve_is_fixed_ipv4_loopback_and_valid_port(initialized, options):
    path, *_ = initialized
    result = RUNNER.invoke(cli.app, ["serve", str(path), "--test-mode", *options])
    assert result.exit_code != 0


def test_serve_mocked_server_uses_port_8005_no_proxy_or_access_logging(initialized, monkeypatch):
    import uvicorn

    from parameter_agent import provider

    path, *_ = initialized
    instance = Mock(fingerprint="TEST-provider")
    factory = Mock(return_value=instance)
    monkeypatch.setattr(provider, "SyntheticProvider", factory)
    server = Mock()
    monkeypatch.setattr(uvicorn, "run", server)
    result = RUNNER.invoke(cli.app, ["serve", str(path), "--test-mode"])
    assert result.exit_code == 0, result.output
    factory.assert_called_once_with()
    instance.decide.assert_not_called()
    instance.close.assert_called_once_with()
    server.assert_called_once()
    assert server.call_args.kwargs == {
        "host": "127.0.0.1", "port": 8005, "access_log": False, "proxy_headers": False,
        "server_header": False, "log_level": "warning"}


def test_serve_mocked_local_url_and_custom_port(initialized, monkeypatch):
    import uvicorn

    from parameter_agent import provider

    path, *_ = initialized
    instance = Mock(fingerprint="local-explicit")
    factory = Mock(return_value=instance)
    monkeypatch.setattr(provider, "LocalProvider", factory)
    server = Mock()
    monkeypatch.setattr(uvicorn, "run", server)
    result = RUNNER.invoke(cli.app, ["serve", str(path), "--model-url", "http://127.0.0.1:9999", "--port", "8015"])
    assert result.exit_code == 0, result.output
    factory.assert_called_once_with("http://127.0.0.1:9999")
    assert server.call_args.kwargs["port"] == 8015
    instance.close.assert_called_once_with()


@pytest.mark.parametrize("url", ["http://localhost:9999", "http://192.168.1.5:9999",
                                 "https://127.0.0.1:9999", "http://127.0.0.1:9999/v1",
                                 "".join(("http://127.0.0.1:9999", "?", "token=x")),
                                 "".join(("http://", "user@", "127.0.0.1:9999")),
                                 "http://127.0.0.1:9999#", "http://[::1]suffix:9999",
                                 "http://127.0.0.1:0", "http://2130706433:9999"])
def test_serve_only_accepts_numeric_loopback_root_url(initialized, monkeypatch, url):
    from parameter_agent import provider

    path, *_ = initialized
    factory = Mock(side_effect=AssertionError("invalid URL must not construct provider"))
    monkeypatch.setattr(provider, "LocalProvider", factory)
    result = RUNNER.invoke(cli.app, ["serve", str(path), "--model-url", url])
    assert result.exit_code != 0
    assert "model_url" in result.output
    factory.assert_not_called()


def test_serve_mocked_managed_lifecycle_and_failure_cleanup(initialized, monkeypatch):
    import uvicorn

    from parameter_agent import managed_provider

    path, *_ = initialized
    instance = Mock(fingerprint="managed-explicit")
    factory = Mock(return_value=instance)
    monkeypatch.setattr(managed_provider, "ManagedLocalProvider", factory)
    server = Mock(side_effect=RuntimeError("secret-internal-text"))
    monkeypatch.setattr(uvicorn, "run", server)
    # This path is passed only to a mock; no runtime config/model is created or opened.
    result = RUNNER.invoke(cli.app, ["serve", str(path), "--runtime-config", "/explicit-not-read.json"])
    assert result.exit_code != 0
    assert "secret-internal-text" not in result.output
    factory.assert_called_once_with(Path("/explicit-not-read.json"))
    instance.close.assert_called_once_with()


def test_serve_rejects_relative_runtime_file_before_constructor(initialized, monkeypatch):
    from parameter_agent import managed_provider

    path, *_ = initialized
    factory = Mock(side_effect=AssertionError("must not open runtime file"))
    monkeypatch.setattr(managed_provider, "ManagedLocalProvider", factory)
    result = RUNNER.invoke(cli.app, ["serve", str(path), "--runtime-config", "not-read.json"])
    assert result.exit_code != 0
    assert "runtime_config" in result.output
    factory.assert_not_called()


def test_main_dispatches_entrypoint_without_server(monkeypatch):
    app = Mock()
    monkeypatch.setattr(cli, "app", app)
    cli.main(["fixture", "--case", "normal"])
    app.assert_called_once_with(args=["fixture", "--case", "normal"], prog_name="parameter-recommendation")
