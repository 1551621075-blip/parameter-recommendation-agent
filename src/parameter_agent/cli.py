"""Explicit local bootstrap and training; credentials never come from arguments or environment."""

import json
import secrets
import time
import uuid
from functools import wraps
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

import typer
from pydantic import ValidationError

from . import bc
from .domain import digest
from .errors import DomainError
from .fixtures import demo_contexts, generate_demonstrations
from .service import RecommendationService
from .storage import Store

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False,
                  help="Local parameter recommendation with original synthetic data only.")
TRAINING_PLANT = "synthetic-plant"
FIXTURE_CASES = ("short_shot", "flash", "warpage", "normal", "missing", "ood")


def _output(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _safe(command):
    @wraps(command)
    def wrapped(*args, **kwargs):
        try:
            return command(*args, **kwargs)
        except DomainError as exc:
            typer.echo(f"{exc.code}: {exc.message}", err=True)
            raise typer.Exit(1) from None
        except ValidationError:
            typer.echo("validation_error: Invalid fields, types or ranges", err=True)
            raise typer.Exit(2) from None
        except (typer.Exit, typer.Abort):
            raise
        except Exception:
            typer.echo("local_error: Local operation failed; internal details withheld", err=True)
            raise typer.Exit(1) from None
    return wrapped


def _safe_path(data: Path) -> Path:
    path = data.expanduser().absolute()
    if ".." in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise DomainError("unsafe_path", "Symlinks and parent traversal are not allowed", 409)
    return path


def _existing_store(data: Path) -> Store:
    path = _safe_path(data)
    if not path.is_dir() or not (path / "data.sqlite3").is_file():
        raise DomainError("not_initialized", "Initialize a new data directory first", 409)
    return Store(path)


def _authenticated(data: Path):
    store = _existing_store(data)
    token = typer.prompt("Bearer token (hidden)", hide_input=True)
    return RecommendationService(store), store.authenticate(token)


@app.command("init")
@_safe
def initialize(data: Path):
    """Create a NEW directory, import original contexts and show one admin token once."""
    path = _safe_path(data)
    if path.exists():
        raise DomainError("init_exists", "Init requires a new, nonexistent directory", 409)
    if not path.parent.is_dir():
        raise DomainError("init_parent", "The parent directory must already exist", 409)
    contexts = demo_contexts(time.time_ns() // 1_000_000)
    plants = sorted({context.plant_id for context in contexts.values()})
    path.mkdir(mode=0o700, exist_ok=False)
    store = Store(path)
    token = secrets.token_urlsafe(32)
    principal = store.create_user("local-admin", token, "admin", plants)
    service = RecommendationService(store)
    for case, context in contexts.items():
        service.import_context(principal, context, action_id="init-" + case)
    _output({"user_id": principal.user_id, "role": principal.role, "plants": plants,
             "token": token, "contexts_imported": len(contexts),
             "notice": "Token shown once; only its hash is stored. Original synthetic data only. "
                       "No model was trained, recommendation submitted, approval made or device run."})


@app.command("fixture")
@_safe
def fixture(case: Annotated[str, typer.Option("--case")] = "short_shot"):
    """Print one original synthetic ProcessContext JSON object without persisting it."""
    if case not in FIXTURE_CASES:
        raise DomainError("unknown_case", "Choose short_shot, flash, warpage, normal, missing or ood", 422)
    _output(demo_contexts(time.time_ns() // 1_000_000)[case])


@app.command("train")
@_safe
def train(data: Path):
    """Actually train BC locally and register safe JSON artifacts; prompt for the token."""
    service, principal = _authenticated(data)
    store = service.store
    store.require(principal, TRAINING_PLANT, roles=("admin", "reviewer"))
    result = bc.train_model(generate_demonstrations())
    model, metrics = result["model"], result["metrics"]
    bc.validate_model(model)
    # Store's content-addressed JSON publisher is atomic and never overwrites artifacts.
    # Authorization is rechecked after expensive training, before publishing or registering.
    store.require(principal, TRAINING_PLANT, roles=("admin", "reviewer"))
    model_artifact = store.put_artifact(model)
    metrics_payload = {"model_hash": model["model_hash"], "metrics": metrics}
    metrics_artifact = store.put_artifact(metrics_payload)
    metrics_id = "metrics-" + digest({"owner": principal.user_id, "plant_id": TRAINING_PLANT,
                                     "model_hash": model["model_hash"]})
    # Publish the bound evaluation before activation. An interrupted activation may leave an
    # honest unreferenced evaluation, but can never associate another model's metrics.
    with store.transaction() as connection:
        store.require(principal, TRAINING_PLANT, roles=("admin", "reviewer"), conn=connection)
        try:
            existing = store.get_object("model_metrics", metrics_id, principal, conn=connection)
        except DomainError as exc:
            if exc.code != "not_found" or exc.status != 404:
                raise
            store.create_object(connection, kind="model_metrics", object_id=metrics_id,
                                principal=principal, plant_id=TRAINING_PLANT, status="evaluated",
                                payload=metrics_payload)
            store.audit(connection, principal, kind="model_metrics", object_id=metrics_id,
                        action="synthetic_bc_evaluated", detail={"model_hash": model["model_hash"]})
        else:
            if existing["content_hash"] != metrics_artifact or existing["status"] != "evaluated":
                raise DomainError("metrics_conflict", "Evaluation identity is already bound")
    registered = service.register_model(principal, model, TRAINING_PLANT, "train-" + uuid.uuid4().hex)
    _output({"model_id": registered["object_id"], "model_hash": model["model_hash"],
             "model_version": registered["version"], "model_artifact": model_artifact,
             "metrics_artifact": metrics_artifact, "metrics": metrics,
             "notice": "Actual behavioral cloning on original synthetic data, not field effectiveness. "
                       "No recommendation, approval or execution was created."})


@app.command("serve")
@_safe
def serve(
    data: Path,
    runtime_config: Annotated[Path | None, typer.Option("--runtime-config")] = None,
    model_url: Annotated[str | None, typer.Option("--model-url")] = None,
    test_mode: Annotated[bool, typer.Option("--test-mode")] = False,
    port: Annotated[int, typer.Option("--port")] = 8005,
):
    """Choose exactly one explicit local provider; bind only 127.0.0.1, never a public host."""
    if sum((runtime_config is not None, model_url is not None, test_mode)) != 1:
        raise DomainError("provider_choice", "Choose exactly one of --runtime-config, --model-url, --test-mode")
    if not 1 <= port <= 65535:
        raise DomainError("listen_address", "Serve requires a port in 1..65535")
    if runtime_config is not None and not runtime_config.is_absolute():
        raise DomainError("runtime_config", "Runtime configuration must be an absolute path")
    # Fail on an uninitialized/unsafe directory before constructing any provider.
    store = _existing_store(data)
    from .api import _authority, create_app
    from .provider import LocalProvider, SyntheticProvider

    if model_url is not None:
        try:
            parsed = urlsplit(model_url)
            if (parsed.scheme != "http" or parsed.path not in {"", "/"}
                    or parsed.query or parsed.fragment or "?" in model_url or "#" in model_url
                    or any(ord(char) <= 32 or ord(char) >= 127 for char in model_url)):
                raise ValueError("Root-only numeric loopback URL required")
            _authority(parsed.netloc, parsed.scheme)
        except ValueError:
            raise DomainError("model_url", "Use an HTTP numeric loopback root URL") from None
    if runtime_config is not None:
        from .managed_provider import ManagedLocalProvider
        provider = ManagedLocalProvider(runtime_config)
    elif model_url is not None:
        provider = LocalProvider(model_url)
    else:
        provider = SyntheticProvider()
    try:
        application = create_app(store.root, provider)
        _output({"provider": provider.fingerprint, "synthetic_only": True,
                 "listen": f"http://127.0.0.1:{port}", "inference_checked": False})
        import uvicorn
        uvicorn.run(application, host="127.0.0.1", port=port, access_log=False,
                    proxy_headers=False, server_header=False, log_level="warning")
    finally:
        provider.close()


def main(argv=None):
    app(args=argv, prog_name="parameter-recommendation")


if __name__ == "__main__":
    main()
