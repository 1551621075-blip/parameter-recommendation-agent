"""Inert browser contract tests: no Chrome/server/model/training/network or background threads."""
from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "browser_smoke.mjs"
# Invoke an explicit installed Node 24 executable, never npx/download/runtime installation.
NODE_CANDIDATES = [
    Path.home() / ".nvm/versions/node/v24.18.0/bin/node",
    Path("/opt/homebrew/opt/node@24/bin/node"),
    Path("/usr/local/opt/node@24/bin/node"),
]
if shutil.which("node"):
    NODE_CANDIDATES.append(Path(shutil.which("node")))


@pytest.fixture(scope="module")
def node24():
    for path in NODE_CANDIDATES:
        if path.is_file():
            result = subprocess.run([str(path), "--version"], capture_output=True, text=True,
                                    timeout=10, check=False)
            if result.returncode == 0 and result.stdout.startswith("v24."):
                return str(path)
    pytest.skip("An explicitly installed Node 24 executable is required; no downloads permitted")


def node(executable, *args):
    return subprocess.run(
        [executable, *map(str, args)], cwd=PROJECT, capture_output=True, text=True,
        timeout=15, check=False,
        env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
    )


@pytest.fixture(scope="module")
def exports(node24):
    result = node(
        node24, "--input-type=module", "-e",
        f"const m=await import({json.dumps(SCRIPT.as_uri())});"
        "console.log(JSON.stringify({python:m.PY_SERVER,selectors:m.SELECTORS,project:m.PROJECT}));",
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.fixture
def bootstrap_definitions(exports):
    # Only pure definitions/imports. The __main__ guard cannot call bootstrap here.
    namespace = {"__name__": "__inert_contract__"}
    exec(compile(exports["python"], "<inert browser definitions>", "exec"), namespace)
    return namespace


def test_node24_syntax_and_safe_default(node24):
    assert node(node24, "--check", SCRIPT).returncode == 0
    for arguments in [[], ["--help"]]:
        result = node(node24, SCRIPT, *arguments)
        assert result.returncode == 0, result.stderr
        assert "No browser starts without --run-chrome" in result.stdout
        assert "Node 24" in result.stdout and "--artifacts-parent" in result.stdout
        assert "synthetic clock" in result.stdout and "NOT 60s wall time" in result.stdout
    result = node(node24, SCRIPT, "--self-test")
    assert result.returncode == 0, result.stderr
    assert "PASS inert" in result.stdout


@pytest.mark.parametrize("arguments", [
    ["--run-chrome"],
    ["--artifacts-parent", "/private/tmp"],
    ["--run-chrome", "--artifacts-parent", str(PROJECT)],
    ["--run-chrome", "--artifacts-parent", str(PROJECT / "scripts")],
    ["--run-chrome", "--artifacts-parent", "relative"],
    ["--run-chrome", "--artifacts-parent", "/private/tmp", "--run-chrome"],
    ["--run-chrome", "--artifacts-parent", "/private/tmp", "--chrome", "relative"],
    ["--run-chrome", "--artifacts-parent", "/private/tmp", "--clock-ms", "60000"],
    ["--model-url", "http://127.0.0.1:8000"],
    ["--token", "forbidden-token"],
    ["--help", "--run-chrome"],
])
def test_invalid_commands_fail_before_launch(node24, arguments):
    result = node(node24, SCRIPT, *arguments)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "Usage:" in result.stderr


def test_import_help_selftest_and_rejected_args_are_inert_even_when_io_is_trapped(node24):
    # Imports may READ the script. All mutation/process/network entrypoints are forbidden.
    expression = r"""
import child from 'node:child_process';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import http from 'node:http';
import https from 'node:https';
import net from 'node:net';
import {syncBuiltinESMExports} from 'node:module';
const forbidden=()=>{throw new Error('NONINERT OPERATION');};
for (const name of ['spawn','exec','execFile','fork','spawnSync','execSync','execFileSync']) child[name]=forbidden;
for (const name of ['writeFile','appendFile','mkdir','mkdtemp','rm','rename','unlink','rmdir','copyFile','truncate']) {
  fs[name]=forbidden; if (name+'Sync' in fs) fs[name+'Sync']=forbidden;
  if (name in fsp) fsp[name]=forbidden;
}
// Node's module loader legitimately opens the source read-only; forbid mutation flags.
for (const [object,name] of [[fs,'open'],[fs,'openSync'],[fsp,'open']]) {
  const original=object[name];
  object[name]=function(path,flags,...rest) {
    if (flags !== 'r' && flags !== fs.constants.O_RDONLY) forbidden();
    return original.call(this,path,flags,...rest);
  };
}
http.get=http.request=http.createServer=https.get=https.request=https.createServer=forbidden;
net.connect=net.createConnection=net.createServer=forbidden;
globalThis.fetch=forbidden; globalThis.WebSocket=class {constructor(){forbidden();}};
syncBuiltinESMExports();
"""
    expression += f"const uri={json.dumps(SCRIPT.as_uri())};"
    expression += "const m=await import(uri);m.selfTest();"
    expression += f"process.argv=[process.execPath,{json.dumps(str(SCRIPT))}];"
    expression += "await import(uri+'?inert-default');"
    expression += "process.argv.push('--help');await import(uri+'?inert-help');"
    expression += "process.argv[2]='--self-test';await import(uri+'?inert-self-test');"
    expression += "process.argv[2]='--run-chrome';await import(uri+'?inert-rejected');"
    result = node(node24, "--input-type=module", "-e", expression)
    assert result.returncode == 2, result.stderr
    assert "PASS inert" in result.stdout and "Usage:" in result.stderr
    assert "NONINERT OPERATION" not in result.stdout + result.stderr


def test_bootstrap_compiles_and_seeds_only_real_inputs_and_actual_training(exports):
    source = exports["python"]
    tree = ast.parse(source)
    compile(tree, "<parameter browser bootstrap>", "exec")
    bootstrap = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                     and item.name == "bootstrap")
    calls = [ast.unparse(item.func) for item in ast.walk(bootstrap) if isinstance(item, ast.Call)]
    assert "train_model(generate_demonstrations())" in source
    assert "demo_contexts(clock())" in source
    assert "create_app(data_root, SyntheticProvider(), clock=clock)" in source
    assert "app.state.service.import_context" in calls
    assert "app.state.service.register_model" in calls
    assert "app.state.store.create_user" in calls
    assert "kind='model_metrics'" in source
    assert "payload={'model_hash': training['model']['model_hash'], 'metrics': training['metrics']}" in source
    assert not any(call.endswith((".review", ".create", ".run", ".readback", ".observe",
                                  ".request_execution", ".apply_synthetic"))
                   for call in calls if call != "server.run")
    assert "server.run(sockets=[sock])" in source
    assert "sock.bind(('127.0.0.1', 0))" in source
    assert "secrets.token_urlsafe(32)" in source
    assert "sys.argv[2]" not in source
    for forbidden in ["spc_agent", "ManagedLocalProvider", "LocalProvider(", "model_dump(",
                      "model_copy(", ".route(", "@app.", "setattr(", "time.sleep("]:
        assert forbidden not in source
    assert isinstance(tree.body[-1], ast.If)
    assert ast.unparse(tree.body[-1].test) == "__name__ == '__main__'"


def test_definitions_do_not_initialize_any_runtime(exports, monkeypatch):
    import socket
    import threading

    def forbidden(*args, **kwargs):
        pytest.fail("Definitions must not initialize server, process, network or thread")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(threading.Thread, "start", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    namespace = {"__name__": "__inert_contract__"}
    exec(compile(exports["python"], "<definitions only>", "exec"), namespace)
    assert callable(namespace["bootstrap"])
    assert "app" not in namespace and "token" not in namespace


def execution(now=1_750_000_000_000, *, status="executed"):
    target = {"temperature": "208", "speed": "50", "pressure": "47", "cooling": "18"}
    return {"kind": "execution", "object_id": "exec-rec-test", "status": status,
            "payload": {"target": target,
                        "readback": {"execution_id": "exec-rec-test", "synthetic": True,
                                     "readback_at_ms": now, "readback_hash": "a" * 64,
                                     "current": dict(target)}}}


def complete(coroutine):
    """Drive an immediately completing fake ASGI coroutine without event loop/socketpair."""
    try:
        coroutine.send(None)
    except StopIteration as end:
        return end.value
    pytest.fail("Inert ASGI test attempted actual asynchronous I/O")


def harness_exchange(namespace, *, path="/api/executions/exec-rec-test/readback",
                     method="POST", status=200, response=None, headers=None,
                     send_fails=False, chunked=False):
    clock = namespace["SyntheticClock"](1_750_000_000_000)
    gate = {"armed": True, "held": False, "released": 0}
    messages, ages, waits = [], [], []
    value = execution() if response is None else response
    body = json.dumps(value, separators=(",", ":")).encode()
    if headers is None:
        headers = [(b"authorization", b"Bearer IN_MEMORY_ONLY")]
    original = [{"type": "http.response.start", "status": status, "headers": []}]
    if chunked:
        original.append({"type": "http.response.body", "body": body[:15], "more_body": True})
        original.append({"type": "http.response.body", "body": body[15:], "more_body": False})
    else:
        original.append({"type": "http.response.body", "body": body, "more_body": False})

    class Release:
        def wait(self, timeout):
            waits.append(timeout)
            assert gate["held"] is True
            return True

    async def app(scope, receive, send):
        assert scope["path"] == path
        for message in original:
            await send(message)

    async def receive():
        pytest.fail("Wrapper must not consume or fabricate request bodies")

    async def reply(message):
        ages.append(clock())
        if send_fails and message["type"] == "http.response.body":
            raise ConnectionError("synthetic send failure")
        messages.append(message)

    wrapper = namespace["SmokeHarness"](app, "IN_MEMORY_ONLY", clock, gate, Release())
    scope = {"type": "http", "method": method, "path": path, "headers": headers}
    if send_fails:
        with pytest.raises(ConnectionError):
            complete(wrapper(scope, receive, reply))
    else:
        complete(wrapper(scope, receive, reply))
        assert messages == original  # exact original real-response messages, never rewritten
    return clock, gate, ages, waits


@pytest.mark.parametrize("chunked", [False, True])
def test_synthetic_clock_advances_only_after_successful_original_readback_send(bootstrap_definitions,
                                                                          chunked):
    clock, _, ages, _ = harness_exchange(bootstrap_definitions, chunked=chunked)
    assert clock.advances == 1
    assert clock() == 1_750_000_060_001
    assert set(ages) == {1_750_000_000_000}
    clock.verified_readback(execution())
    assert clock.advances == 1  # receipt replay cannot advance time twice


@pytest.mark.parametrize("kwargs", [
    {"status": 409},
    {"method": "GET"},
    {"path": "/api/executions/exec-rec-test/apply"},
    {"path": "/api/recommendations/rec-test/observe"},
    {"headers": []},
    {"headers": [(b"authorization", b"Bearer DIFFERENT")]},
    {"headers": [(b"authorization", b"Bearer IN_MEMORY_ONLY")] * 2},
    {"response": execution(status="unknown")},
    {"response": execution(status="applied")},
    {"send_fails": True},
])
def test_failed_untrusted_or_nonreadback_traffic_never_advances_clock(bootstrap_definitions, kwargs):
    clock, _, _, _ = harness_exchange(bootstrap_definitions, **kwargs)
    assert clock.advances == 0 and clock() == 1_750_000_000_000


@pytest.mark.parametrize("changed", [
    {"synthetic": False}, {"current": {"temperature": "999"}},
    {"readback_at_ms": 1}, {"readback_at_ms": True},
    {"readback_hash": "short"}, {"execution_id": "different-execution"},
])
def test_clock_rejects_incomplete_or_mismatched_readback(bootstrap_definitions, changed):
    response = execution()
    response["payload"]["readback"].update(changed)
    clock, _, _, _ = harness_exchange(bootstrap_definitions, response=response)
    assert clock.advances == 0


@pytest.mark.parametrize("advance", [60_000, 61_000, 120_000])
def test_clock_interval_is_explicitly_injectable_for_inert_tests(bootstrap_definitions, advance):
    clock = bootstrap_definitions["SyntheticClock"](1_750_000_000_000, advance_ms=advance)
    clock.verified_readback(execution())
    assert clock() == 1_750_000_000_000 + advance


@pytest.mark.parametrize("advance", [0, 59_999, 120_001, True, 60_000.0])
def test_clock_rejects_unsafe_intervals(bootstrap_definitions, advance):
    with pytest.raises(ValueError):
        bootstrap_definitions["SyntheticClock"](1, advance_ms=advance)


def test_delayed_logout_response_is_real_get_only_and_uses_no_control_endpoint(bootstrap_definitions,
                                                                           monkeypatch):
    async def inline_thread(function, timeout):
        return function(timeout)

    monkeypatch.setattr(bootstrap_definitions["asyncio"], "to_thread", inline_thread)
    clock, gate, _, waits = harness_exchange(bootstrap_definitions, path="/api/contexts", method="GET")
    assert waits == [15] and gate == {"armed": False, "held": False, "released": 1}
    assert clock.advances == 0
    _, gate, _, waits = harness_exchange(bootstrap_definitions, path="/api/contexts", method="POST")
    assert waits == [] and gate["armed"] is True


def test_factory_clock_injection_and_no_production_clock_backdoor(exports):
    source = (PROJECT / "src/parameter_agent/api.py").read_text()
    tree = ast.parse(source)
    factory = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                   and item.name == "create_app")
    assert "clock" in [item.arg for item in factory.args.kwonlyargs]
    assert "RecommendationService(store, clock=clock)" in source
    assert "ActionService(service, clock=clock)" in source
    assert "AgentEngine(service, provider, clock=clock)" in source
    assert "SmokeHarness" not in source and "SyntheticClock" not in source
    assert "clock.verified_readback(json.loads(body))" in exports["python"]


def test_selectors_match_actual_shipped_ui(exports):
    html = (PROJECT / "src/parameter_agent/templates/index.html").read_text()
    for selector in exports["selectors"].values():
        for identifier in re.findall(r"#([a-z][a-z0-9-]*)", selector):
            assert f'id="{identifier}"' in html, selector
    javascript = (PROJECT / "src/parameter_agent/static/app.js").read_text()
    for marker in ["candidate.raw_delta", "candidate.target", "candidate.delta", "candidate.directions",
                   'credentials: "omit"', 'cache: "no-store"', "crypto.randomUUID()", "controller.abort()"]:
        assert marker in javascript


def test_explicit_review_options_are_real_html_tags():
    class Options(HTMLParser):
        selected = False

        def __init__(self):
            super().__init__()
            self.values = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "select":
                self.selected = attrs.get("id") == "review-decision"
            elif tag == "option" and self.selected:
                self.values.append(attrs.get("value"))

        def handle_endtag(self, tag):
            if tag == "select":
                self.selected = False

    parser = Options()
    parser.feed((PROJECT / "src/parameter_agent/templates/index.html").read_text())
    assert parser.values == ["", "approved", "rejected"]


def test_no_unowned_profiles_process_kills_or_artifact_payloads():
    source = SCRIPT.read_text()
    for forbidden in ["killall", "pkill", "--no-sandbox", "--disable-web-security", "puppeteer",
                      "playwright", "Application Support/Google/Chrome", "process.env", "shell: true",
                      "ManagedLocalProvider", "Page.captureScreenshot", "writeFileSync", "seriesRuns"]:
        assert forbidden not in source
    for required in ["--user-data-dir=${profile}", "await realpath(this.options.parent)",
                     "mkdtemp(join(parent, 'parameter-browser-'))", "ownedDirectory(this.runtime)",
                     "row.started === owned.started", "constants.O_NOFOLLOW", "Network.getResponseBody",
                     "Input.dispatchMouseEvent", "Input.insertText", "no_plaintext_tokens_in_owned_files",
                     "synthetic clock", "wall_wait_performed: false", "physical_evidence: false"]:
        assert required in source
    assert source.count("await writeFile(") == 1
    assert "await writeFile(join(this.root, 'summary.json'), output, {flag: 'wx', mode: 0o600})" in source
    assert "this.requestBody(response, ['action_id'])" in source  # no fault/seed/sample-count controls


def test_child_environment_is_minimal_and_token_free(node24):
    result = node(
        node24, "--input-type=module", "-e",
        f"const m=await import({json.dumps(SCRIPT.as_uri())});const s=new m.Smoke({{}});"
        "s.runtime='/private/tmp/OWNED';s.secrets=['not-an-environment-variable'];"
        "console.log(JSON.stringify(s.environment()));",
    )
    assert result.returncode == 0, result.stderr
    environment = json.loads(result.stdout)
    assert set(environment) == {"PATH", "HOME", "TMPDIR", "LANG", "PYTHONPATH",
                                "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE"}
    assert environment["HOME"] == environment["TMPDIR"] == "/private/tmp/OWNED"
    assert environment["PYTHONNOUSERSITE"] == environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "token" not in result.stdout.lower()


def test_descendant_tracking_survives_reparenting_and_rejects_pid_reuse(node24):
    result = node(
        node24, "--input-type=module", "-e",
        f"const m=await import({json.dumps(SCRIPT.as_uri())});const s=new m.Smoke({{}});"
        "const child={pid:100,exitCode:null,signalCode:null};s.children=[child];"
        "s.processTable=async()=>[{pid:100,parent:1,started:'A'},{pid:101,parent:100,started:'B'},"
        "{pid:102,parent:101,started:'C'},{pid:200,parent:1,started:'UNOWNED'}];"
        "const initial=await s.snapshotOwnedProcesses();child.exitCode=1;"
        "s.processTable=async()=>[{pid:101,parent:1,started:'REUSED'},{pid:102,parent:1,started:'C'},"
        "{pid:200,parent:1,started:'UNOWNED'}];"
        "const after=await s.snapshotOwnedProcesses();"
        "console.log(JSON.stringify({initial:initial.map(r=>r.pid),after:after.map(r=>r.pid)}));",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"initial": [100, 101, 102], "after": [102]}


def test_aborted_401_response_retains_observed_status_without_fabricating_body(node24):
    result = node(
        node24, "--input-type=module", "-e",
        f"const m=await import({json.dumps(SCRIPT.as_uri())});const s=new m.Smoke({{}});"
        "s.sessionId='OWNED';s.responses=[{requestId:'r',status:401,done:false}];"
        "await s.observe({sessionId:'OWNED',method:'Network.loadingFailed',params:{requestId:'r'}});"
        "console.log(JSON.stringify(s.responses));",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        {"requestId": "r", "status": 401, "done": True, "body": None, "aborted": True}
    ]


def test_safe_errors_redact_tokens_and_never_dump_compared_payloads(node24):
    result = node(
        node24, "--input-type=module", "-e",
        f"const m=await import({json.dumps(SCRIPT.as_uri())});const s=new m.Smoke({{}});"
        "s.secrets=['SECRET'];console.log(JSON.stringify(s.safe('SECRET mismatch\\n+ PRIVATE PAYLOAD')));",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == "[REDACTED] mismatch"
