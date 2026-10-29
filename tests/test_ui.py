"""Static and native-DOM contract checks. No browser, server, network or runtime data."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "src/parameter_agent/templates/index.html"
JS = ROOT / "src/parameter_agent/static/app.js"
CSS = ROOT / "src/parameter_agent/static/style.css"
NODE = shutil.which("node") or "node"
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


class WorkbenchParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.elements = []
        self.ids = {}
        self.options = {}
        self.errors = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        element = {"tag": tag, "attrs": attrs}
        form = next((item for item in reversed(self.stack) if item["tag"] == "form"), None)
        element["form"] = form["attrs"].get("id") if form else None
        self.elements.append(element)
        if "id" in attrs:
            if attrs["id"] in self.ids:
                self.errors.append(f"Duplicate id {attrs['id']}")
            self.ids[attrs["id"]] = element
        if tag == "option":
            if not self.stack or self.stack[-1]["tag"] != "select":
                self.errors.append("option without an opening select / nested option")
            else:
                self.options.setdefault(self.stack[-1]["attrs"]["id"], []).append(element)
        if tag not in VOID:
            self.stack.append(element)

    def handle_endtag(self, tag):
        if tag in VOID:
            self.errors.append(f"Closing void element {tag}")
        elif not self.stack or self.stack[-1]["tag"] != tag:
            self.errors.append(f"Unmatched closing tag {tag}")
        else:
            self.stack.pop()


@pytest.fixture
def document():
    parser = WorkbenchParser()
    parser.feed(HTML.read_text())
    parser.close()
    assert not parser.errors, parser.errors
    assert not parser.stack, parser.stack
    return parser


def test_valid_opening_options_and_explicit_empty_decision(document):
    """Regression: malformed opening option tags must not silently select approval."""
    selects = {item["attrs"]["id"] for item in document.elements if item["tag"] == "select"}
    assert selects == {
        "context-select", "task-select", "recommendation-select", "review-decision",
        "execution-select",
    }
    for select in selects:
        assert document.options[select][0]["attrs"]["value"] == ""
    decisions = document.options["review-decision"]
    assert [entry["attrs"]["value"] for entry in decisions] == ["", "approved", "rejected"]
    assert not any("selected" in entry["attrs"] for entry in decisions[1:])
    for identifier in ("execution-confirm", "apply-confirm", "reconcile-confirm"):
        attrs = document.ids[identifier]["attrs"]
        assert attrs["type"] == "checkbox"
        assert "checked" not in attrs
        assert "required" in attrs
    for identifier in ("review-note", "supplement-note", "reconcile-note"):
        assert "required" in document.ids[identifier]["attrs"]
        assert document.ids[identifier]["attrs"]["maxlength"] == "2000"


def test_local_accessible_native_dom_and_complete_selectors(document):
    text = HTML.read_text()
    assert '<html lang="zh-CN">' in text
    assert "NOT production" in text and "绝非生产系统" in text
    assert "配对模拟比较" in text and "至少 60 秒" in text
    for item in document.elements:
        attrs = item["attrs"]
        assert not any(key.lower().startswith("on") for key in attrs)
        if item["tag"] in {"script", "link"}:
            assert (attrs.get("src") or attrs.get("href")).startswith("/static/")
        if item["tag"] == "script":
            assert "defer" in attrs
        if item["tag"] == "label" and "for" in attrs:
            assert attrs["for"] in document.ids
        if item["tag"] == "a" and attrs.get("href", "").startswith("#"):
            assert attrs["href"][1:] in document.ids
    refs = set(re.findall(r'\$\("([a-z][a-z0-9-]*)"\)', JS.read_text()))
    assert refs <= document.ids.keys()
    assert document.ids["workbench"]["attrs"]["aria-busy"] == "false"
    assert document.ids["error-message"]["attrs"]["role"] == "alert"
    assert document.ids["message"]["attrs"]["aria-live"] == "polite"
    assert document.ids["token"]["attrs"]["type"] == "password"
    assert "[hidden] { display: none !important; }" in CSS.read_text()
    assert "@media" in CSS.read_text()


def test_security_invariants_and_javascript_syntax():
    script = JS.read_text()
    for forbidden in (
        "innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(",
        "new Function", "localStorage", "sessionStorage", "indexedDB", "document.cookie",
        "sendBeacon", "WebSocket", "XMLHttpRequest",
    ):
        assert forbidden not in script
    assert 'credentials: "omit"' in script
    assert 'cache: "no-store"' in script
    assert 'redirect: "error"' in script
    assert "crypto.randomUUID()" in script
    assert "new AbortController()" in script
    assert "epoch += 1" in script and "expectedEpoch !== epoch" in script
    assert 'window.addEventListener("pagehide"' in script
    assert ".textContent" in script and "JSON.stringify(body)" in script
    assert "JSON.stringify(result)" not in script
    assert not re.search(r"https?://|@import", HTML.read_text() + CSS.read_text() + script)
    subprocess.run([NODE, "--check", str(JS)], check=True, capture_output=True, text=True)


# A small native-DOM double runs the actual app.js, not a copied implementation.
# Fetch below is in-memory and records exact requests; no runtime profiles or sockets.
DOM_HARNESS = r'''
const assert = require("node:assert/strict");
const vm = require("node:vm");
const crypto = require("node:crypto");
const input = JSON.parse(require("node:fs").readFileSync(0, "utf8"));
const ids = new Map();
class Element {
  constructor(tag, attrs = {}) {
    this.tagName = tag.toUpperCase(); this.nodeType = 1;
    this.attrs = attrs; this.id = attrs.id || ""; this.children = []; this.listeners = {};
    this.dataset = {}; this.className = ""; this.classList = {remove() {}, add() {}};
    this.hidden = "hidden" in attrs; this.disabled = "disabled" in attrs;
    this.required = "required" in attrs; this.value = attrs.value || "";
    this.checked = "checked" in attrs; this._text = "";
  }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  getAttribute(name) { return this.attrs[name] ?? null; }
  set textContent(text) { this._text = String(text); this.children = []; }
  get textContent() { return this._text + this.children.map(c => c.textContent || "").join(" "); }
  append(...items) { this.children.push(...items); }
  prepend(...items) { this.children.unshift(...items); }
  replaceChildren(...items) { this._text = ""; this.children = [...items]; }
  addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
  reset() {
    for (const entry of input.elements.filter(e => e.form === this.id)) {
      const target = ids.get(entry.attrs.id);
      if (target) { target.value = entry.attrs.value || ""; target.checked = "checked" in entry.attrs; }
    }
  }
  reportValidity() {
    return input.elements.filter(e => e.form === this.id).every(e => {
      const item = ids.get(e.attrs.id);
      return !item || item.disabled || !item.required ||
        (e.attrs.type === "checkbox" ? item.checked : !!item.value);
    });
  }
}
for (const entry of input.elements) if (entry.attrs.id) ids.set(entry.attrs.id, new Element(entry.tag, entry.attrs));
const $ = id => { assert(ids.has(id), `Unknown DOM id: ${id}`); return ids.get(id); };
const windowListeners = {};
const requests = [];
let intercept = null;
let role = "admin";
let failObserve = false;
let failApply = false;
let serverNow = Date.now();
const clone = value => JSON.parse(JSON.stringify(value));
const hash = value => crypto.createHash("sha256").update(JSON.stringify(value)).digest("hex");
function obj(kind, id, status, p, version = 1) {
  return {kind, object_id: id, owner: "synthetic-admin", plant_id: "synthetic-plant", status,
    version, content_hash: hash([id, status, p, version]), payload: p};
}
const current = {temperature: "210", speed: "50", pressure: "45", cooling: "15"};
const units = {temperature: "degC", speed: "mm/s", pressure: "MPa", cooling: "s"};
const contextPayload = (id, missing = false) => ({context_id: id, plant_id: "synthetic-plant",
  machine_id: `device-${id}`, material_id: "synthetic-polymer", mold_id: "synthetic-mold",
  process_version: "synthetic-molding-v1", revision: 1, observed_at_ms: Date.now(), defect: "short_shot",
  severity: 0.7, current, units, material_viscosity: missing ? null : 1.1,
  ambient_c: missing ? null : 22.0, sample_count: 120, measurement_ok: true, synthetic: true});
const state = {
  contexts: [obj("context", "ctx-a", "active", contextPayload("ctx-a")),
    obj("context", "ctx-b", "active", contextPayload("ctx-b")),
    obj("context", "ctx-missing", "active", contextPayload("ctx-missing", true))],
  models: [obj("model", "active-model-synthetic-plant", "active", {model_hash: "b".repeat(64),
    metrics: {synthetic: true, training: {initial_loss: 2.4, final_loss: 0.3},
      test: {combination_accuracy: 0.83, delta_mae: {temperature: 1.2}, no_action: 0.4},
      split: {train: {groups: ["train-a"]}, calibration: {groups: ["cal-a"]}, test: {groups: ["test-a"]}}}})],
  tasks: [], recommendations: [], executions: [], audit: []
};
function task(id, ctx, status = "created", extra = {}) {
  const context = state.contexts.find(c => c.object_id === ctx);
  return obj("task", id, status, {context_id: ctx,
    source_snapshot: {context_hash: context.content_hash, context_version: context.version,
      model_hash: "b".repeat(64), model_version: 1, constraint_version: "synthetic-bop-v1"},
    evidence: [], prediction: null, candidate: null, status, calls: 0, tools: 0, tokens: 0,
    elapsed_s: 0, error: null, pending_action: null, awaiting_fields: [], lease_id: null,
    unknown_calls: 0, lease_until_ms: 0, ...extra});
}
state.tasks = [task("task-a", "ctx-a"), task("task-b", "ctx-b"),
  task("task-missing", "ctx-missing", "awaiting_input", {awaiting_fields: ["material_viscosity", "ambient_c"]})];
function replace(kind, value) {
  const index = state[kind].findIndex(item => item.object_id === value.object_id);
  if (index === -1) state[kind].push(value); else state[kind][index] = value;
  return value;
}
function update(kind, entry, status, p = entry.payload) {
  return replace(kind, obj(entry.kind, entry.object_id, status, p, entry.version + 1));
}
function recommendation(taskEntry) {
  const ctx = state.contexts.find(c => c.object_id === taskEntry.payload.context_id);
  const raw_delta = {temperature: "2.50", speed: "0", pressure: "0", cooling: "0"};
  const candidate = {eligible: true, issues: [], raw_delta, delta: {...raw_delta, temperature: "2"},
    target: {...current, temperature: "212"}, directions: {temperature: "increase", speed: "hold", pressure: "hold", cooling: "hold"},
    context_hash: ctx.content_hash, context_id: ctx.object_id, model_hash: "b".repeat(64),
    constraint_version: "synthetic-bop-v1", candidate_hash: "c".repeat(64)};
  const evidence = [{evidence_id: "ev-ctx", task_id: taskEntry.object_id, tool: "get_process_context",
    context_hash: ctx.content_hash, content_hash: "d".repeat(64), payload: {context: ctx.payload}},
    {evidence_id: "ev-bc", task_id: taskEntry.object_id, tool: "predict_bc", context_hash: ctx.content_hash,
      content_hash: "e".repeat(64), payload: {status: "ok", confidence: 0.85, raw_delta}},
    {evidence_id: "ev-exp", task_id: taskEntry.object_id, tool: "retrieve_experience", context_hash: ctx.content_hash,
      content_hash: "f".repeat(64), payload: {items: [{id: "original-1", text: "原创合成经验，不能豁免约束", provenance: "synthetic-experience-v1"}]}}];
  const rec = obj("recommendation", `rec-${taskEntry.object_id}`, "pending_review", {
    task_id: taskEntry.object_id, context_id: ctx.object_id, context_hash: ctx.content_hash,
    context_version: ctx.version, model_hash: "b".repeat(64), model_version: 1,
    constraint_version: "synthetic-bop-v1", candidate, evidence, synthetic: true,
    untrusted_model_draft: '<img src=x onerror="attack()">这只是未审核解释'});
  replace("recommendations", rec);
  update("tasks", taskEntry, "submitted", {...taskEntry.payload, status: "submitted", candidate, evidence});
  return rec;
}
function response(value, code = 200) { return {ok: code < 400, status: code, json: async () => clone(value)}; }
function server(path, init, body) {
  if (path === "/health") return response({status: "ok", provider: "TEST · original synthetic", synthetic_only: true});
  if (path === "/api/me") return response({user_id: "synthetic-admin", role, plants: ["synthetic-plant"], acl_version: 1});
  const [kind, id, action] = path.replace(/^\/api\//, "").split("/");
  const rows = state[kind];
  assert(rows, `Unknown endpoint: ${path}`);
  if (init.method === "GET") {
    if (!id) return response({items: rows});
    const value = rows.find(item => item.object_id === id);
    return value ? response(value) : response({code: "not_found", message: "对象不存在"}, 404);
  }
  if (kind === "tasks" && !id) {
    assert.deepEqual(Object.keys(body).sort(), ["action_id", "context_id"]);
    return response(replace("tasks", task("task-new", body.context_id)));
  }
  let value = rows.find(item => item.object_id === id);
  assert(value, `Missing fixture: ${path}`);
  if (action === "run") {
    assert.deepEqual(body, {});
    if (value.payload.lease_id && value.payload.lease_until_ms > serverNow) {
      return response({error: {code: "busy", message: "服务端租约仍活跃"}}, 409);
    }
    if (value.payload.unknown_calls > 0) {
      return response(update(kind, value, "blocked", {...value.payload, status: "blocked",
        error: "model_unknown", lease_id: null, lease_until_ms: 0}));
    }
    recommendation(value);
    return response(state.tasks.find(t => t.object_id === id));
  }
  if (action === "cancel") return response(update(kind, value, "cancelled", {...value.payload, status: "cancelled"}));
  if (action === "supplement") {
    assert.deepEqual(Object.keys(body).sort(), ["action_id", "fields", "note"]);
    assert.deepEqual(Object.keys(body.fields).sort(), ["ambient_c", "material_viscosity"]);
    assert.equal(typeof body.fields.ambient_c, "number");
    assert(body.note.trim());
    const ctx = state.contexts.find(c => c.object_id === value.payload.context_id);
    const changed = update("contexts", ctx, "active", {...ctx.payload, ...body.fields, revision: 2});
    return response(update(kind, value, "created", {...value.payload, status: "created", awaiting_fields: [],
      evidence: [], candidate: null, source_snapshot: {...value.payload.source_snapshot, context_hash: changed.content_hash, context_version: changed.version}}));
  }
  if (action === "review") {
    assert.deepEqual(Object.keys(body).sort(), ["action_id", "decision", "note", "recommendation_hash"]);
    assert.equal(body.recommendation_hash, value.content_hash);
    assert(["approved", "rejected"].includes(body.decision)); assert(body.note.trim());
    return response(update(kind, value, body.decision, {...value.payload, approval: {decision: body.decision, note: body.note, synthetic: true}}));
  }
  if (action === "request-execution") {
    assert.equal(value.status, "approved"); assert.equal(body.confirmed, true);
    assert.equal(body.recommendation_hash, value.content_hash);
    assert.deepEqual(Object.keys(body).sort(), ["action_id", "confirmed", "recommendation_hash"]);
    update(kind, value, "execution_pending");
    return response(replace("executions", obj("execution", `exec-${id}`, "pending", {
      recommendation_id: id, recommendation_hash: body.recommendation_hash, before: current,
      target: value.payload.candidate.target, request_id: `exec-${id}`, synthetic: true})));
  }
  if (action === "apply") {
    assert.equal(value.status, "pending");
    const changed = update(kind, value, failApply ? "unknown" : "applied", {...value.payload,
      device_receipt: {device_revision: 2, target: value.payload.target, synthetic: true}});
    if (failApply) throw new TypeError("ACK lost after synthetic update");
    return response(changed);
  }
  if (action === "readback") {
    assert(["applied", "unknown"].includes(value.status));
    const rec = state.recommendations.find(r => r.object_id === value.payload.recommendation_id);
    update("recommendations", rec, "executed");
    return response(update(kind, value, "executed", {...value.payload,
      readback: {readback_at_ms: Date.now(), readback_hash: "a".repeat(64), current: value.payload.target}}));
  }
  if (action === "observe") {
    assert.equal(value.status, "executed");
    if (failObserve) return response({error: {code: "observation_immature", message: "有效回读后不足 60 秒"}}, 409);
    const observation = {synthetic: true, before: {defect_count: 12, sample_count: 120, defect_rate: 0.1},
      after: {defect_count: 10, sample_count: 120, defect_rate: 10 / 120},
      difference_after_minus_before: -2 / 120, difference_ci95: [-0.1, 0.07],
      conclusion: "no_clear_improvement", causal_effect_proven: false, baseline_is_observed_production: false};
    return response(update(kind, value, "observed", {...value.payload, observation}));
  }
  if (action === "reconcile") {
    assert.equal(role, "admin");
    if (value.payload.lease_id && value.payload.lease_until_ms > serverNow) {
      return response({error: {code: "busy", message: "服务端租约仍活跃，不能对账"}}, 409);
    }
    assert(value.payload.unknown_calls > 0);
    const next = value.status === "cancelled" ? "cancelled" : "created";
    return response(update(kind, value, next, {...value.payload, status: next, unknown_calls: 0,
      error: null, lease_id: null, lease_until_ms: 0}));
  }
  throw new Error(`Unhandled mutation: ${path}`);
}
async function fetchMock(path, init) {
  assert.equal(init.credentials, "omit"); assert.equal(init.cache, "no-store"); assert.equal(init.redirect, "error");
  assert.equal(init.headers.Authorization, ["Bearer", "memory-only-test-token"].join(" "));
  assert(!path.includes("token")); assert(init.signal);
  const body = init.body ? JSON.parse(init.body) : undefined;
  const record = {path, init, body}; requests.push(record);
  if (body && !path.endsWith("/run")) {
    assert.match(body.action_id, /^[a-f0-9-]{36}$/);
    for (const forbidden of ["token", "role", "owner", "seed", "sample_count", "fault", "endpoint"]) assert(!(forbidden in body));
  }
  if (intercept) {
    const result = intercept(record);
    if (result !== undefined) return await result;
  }
  return server(path, init, body);
}
const sandbox = {document: {getElementById: $, createElement: tag => new Element(tag)},
  window: {addEventListener: (name, fn) => {windowListeners[name] = fn;}},
  crypto: {randomUUID: () => crypto.randomUUID()}, fetch: fetchMock, AbortController, DOMException,
  console, Date, setTimeout, clearTimeout};
vm.runInNewContext(input.script, sandbox, {filename: "app.js"});
const tick = async () => { for (let i = 0; i < 5; i++) await new Promise(resolve => setImmediate(resolve)); };
async function emit(id, name) {
  const callbacks = $(id).listeners[name] || [];
  await Promise.all(callbacks.map(fn => fn({preventDefault() {}})));
  await tick();
}
async function choose(id, value) { $(id).value = value; await emit(id, "change"); }
async function fill(id, value) { $(id).value = value; await emit(id, "input"); }
async function check(id) { $(id).checked = true; await emit(id, "change"); }
const mutations = () => requests.filter(r => r.init.method === "POST");
async function login() {
  $("token").value = "memory-only-test-token"; await emit("login-form", "submit");
  assert.equal($("token").value, ""); assert.equal($("workbench").hidden, false);
  assert.equal($("context-select").value, "");
}
async function openSubmitted() {
  recommendation(state.tasks[0]); await login(); await choose("task-select", "task-a");
  assert.equal($("recommendation-select").value, "rec-task-a");
}
async function approve() {
  await choose("review-decision", "approved"); await fill("review-note", "人工核对合成证据与硬约束");
  await emit("review-form", "submit");
}
async function pendingExecution() {
  await openSubmitted(); await approve(); await check("execution-confirm");
  await emit("request-execution-form", "submit");
  assert.equal($("execution-select").value, "exec-rec-task-a");
}
(async () => {
  await (async () => { SCENARIO })();
  console.log("native DOM contract scenario passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
'''


def run_dom(document, scenario):
    result = subprocess.run(
        [NODE, "-e", DOM_HARNESS.replace("SCENARIO", scenario)],
        input=json.dumps({"elements": document.elements, "script": JS.read_text()}),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_task_create_run_and_readable_evidence(document):
    run_dom(document, r'''
      await login();
      assert.equal(mutations().length, 0);
      await choose("context-select", "ctx-a");
      assert.match($("context-parameters").textContent, /degC/);
      assert.match($("context-parameters").textContent, /mm\/s/);
      assert.match($("context-parameters").textContent, /MPa/);
      assert.match($("model-summary").textContent, /0.83/);
      await emit("create-task", "click");
      assert.equal(mutations().length, 1);
      assert.equal($("task-select").value, "task-new");
      assert.equal($("run-task").disabled, false);
      await emit("run-task", "click");
      assert.equal(mutations().length, 2);
      assert.deepEqual(mutations()[1].body, {});
      assert.equal($("review-decision").value, "");
      assert.equal($("review-submit").disabled, true);
      assert.match($("task-candidate").textContent, /2.50/);
      assert.match($("task-candidate").textContent, /212/);
      assert.match($("task-evidence").textContent, /原创合成经验/);
      assert.match($("model-draft").textContent, /<img src=x/);
      assert.equal($("model-draft").children.length, 0);
      assert.equal($("request-execution").disabled, true);
    ''')


def test_task_resume_rebinds_context_even_when_hash_unchanged(document):
    run_dom(document, r'''
      await login();
      await choose("task-select", "task-a");
      const first = $("task-summary").textContent;
      await choose("context-select", "ctx-b");
      assert.equal($("task-select").value, "");
      await choose("task-select", "task-a");
      assert.equal($("context-select").value, "ctx-a");
      assert.equal($("task-summary").textContent, first);
      await choose("task-select", "task-b");
      assert.equal($("context-select").value, "ctx-b");
      await choose("task-select", "task-a");
      assert.equal($("context-select").value, "ctx-a");
      assert.equal(mutations().length, 0);
    ''')


def test_supplement_is_explicit_numeric_and_does_not_run(document):
    run_dom(document, r'''
      await login(); await choose("task-select", "task-missing");
      assert.equal($("context-select").value, "ctx-missing");
      assert.equal($("supplement-form").hidden, false);
      assert.equal($("material-viscosity").disabled, false);
      assert.equal($("ambient-c").disabled, false);
      await fill("material-viscosity", "1.2"); await fill("ambient-c", "23");
      await fill("supplement-note", "人工测量合成数据来源");
      await emit("supplement-form", "submit");
      assert.equal(mutations().length, 1);
      assert.equal(mutations()[0].body.fields.material_viscosity, 1.2);
      assert.equal($("supplement-form").hidden, true);
      assert.equal($("run-task").disabled, false);
      assert.match($("context-summary").textContent, /1.2/);
      assert.match($("task-evidence").textContent, /尚无/);
    ''')


@pytest.mark.parametrize("decision", ["approved", "rejected"])
def test_review_requires_explicit_decision_note_and_current_hash(document, decision):
    run_dom(document, r'''
      await openSubmitted();
      assert.equal($("review-decision").value, "");
      await emit("review-form", "submit"); assert.equal(mutations().length, 0);
      await choose("review-decision", "DECISION");
      await fill("review-note", "   "); assert.equal($("review-submit").disabled, true);
      await fill("review-note", "核对过原始增量与硬约束");
      const beforeHash = state.recommendations[0].content_hash;
      await emit("review-form", "submit");
      assert.equal(mutations().length, 1);
      assert.equal(mutations()[0].body.decision, "DECISION");
      assert.equal(mutations()[0].body.recommendation_hash, beforeHash);
      assert.notEqual(state.recommendations[0].content_hash, beforeHash);
      assert.equal($("review-decision").value, "");
      assert.equal($("execution-confirm").checked, false);
      assert.equal($("request-execution").disabled, true);
      assert.equal(state.executions.length, 0);
    '''.replace("DECISION", decision))


def test_operator_cannot_review(document):
    run_dom(document, r'''
      role = "operator"; await openSubmitted();
      assert.equal($("review-decision").disabled, true);
      assert.equal($("review-note").disabled, true);
      assert.equal($("review-submit").disabled, true);
      await emit("review-form", "submit"); assert.equal(mutations().length, 0);
    ''')


def test_request_apply_readback_observe_are_separate_and_honest(document):
    run_dom(document, r'''
      await pendingExecution();
      assert.equal(mutations().length, 2);
      assert.equal($("apply-confirm").checked, false);
      assert.equal($("apply-execution").disabled, true);
      assert.equal($("readback-execution").disabled, true);
      await check("apply-confirm"); await emit("apply-form", "submit");
      assert.equal(mutations().length, 3);
      assert.equal(state.executions[0].status, "applied");
      assert.match($("execution-detail").textContent, /尚未回读/);
      assert.equal($("readback-execution").disabled, false);
      assert.equal($("observe-recommendation").disabled, true);
      await emit("readback-execution", "click");
      assert.equal(mutations().length, 4);
      assert.equal(state.executions[0].status, "executed");
      assert.equal($("observe-recommendation").disabled, false);
      assert.match($("observation-summary").textContent, /尚无成熟观察/);
      failObserve = true; await emit("observe-recommendation", "click");
      assert.equal($("error-message").hidden, false);
      assert.match($("error-message").textContent, /observation_immature/);
      assert.match($("observation-summary").textContent, /尚无成熟观察/);
      assert.equal(state.recommendations[0].status, "executed");
      failObserve = false; await emit("reload-button", "click");
      await emit("observe-recommendation", "click");
      assert.match($("observation-summary").textContent, /没有明确改善/);
      assert.match($("observation-summary").textContent, /12/);
      assert.match($("observation-summary").textContent, /120/);
      assert.match($("observation-summary").textContent, /95%/);
      assert.match($("observation-summary").textContent, /不证明现场因果/);
      for (const call of mutations().filter(r => /\/(apply|readback|observe)$/.test(r.path))) {
        assert.deepEqual(Object.keys(call.body), ["action_id"]);
      }
      assert.equal(new Set(mutations().map(r => r.body.action_id)).size, mutations().length);
    ''')


def test_lost_apply_ack_locks_retry_until_reload_then_readback(document):
    run_dom(document, r'''
      await pendingExecution(); failApply = true;
      await check("apply-confirm"); await emit("apply-form", "submit");
      assert.equal(state.executions[0].status, "unknown");
      assert.equal($("apply-execution").disabled, true);
      assert.equal($("apply-confirm").checked, false);
      const count = mutations().length;
      await emit("apply-form", "submit"); assert.equal(mutations().length, count);
      await emit("reload-button", "click");
      assert.equal($("apply-execution").disabled, true);
      assert.equal($("readback-execution").disabled, false);
      assert.match($("execution-detail").textContent, /禁止再次应用/);
      await emit("readback-execution", "click");
      assert.equal(state.executions[0].status, "executed");
    ''')


def test_busy_observable_covers_login_reload_and_detail(document):
    run_dom(document, r'''
      const isBusy = () => $("workbench").getAttribute("aria-busy");
      assert.equal(isBusy(), "false");
      let finish;
      intercept = record => record.path === "/api/me" ? new Promise(resolve => {
        finish = () => resolve(server(record.path, record.init, record.body));
      }) : undefined;
      const loggingIn = login(); await tick();
      assert.equal(isBusy(), "true");
      finish(); await loggingIn;
      assert.equal(isBusy(), "false");
      const reloading = emit("reload-button", "click"); await tick();
      assert.equal(isBusy(), "true");
      finish(); await reloading;
      assert.equal(isBusy(), "false");
      intercept = record => record.path === "/api/tasks/task-a" ? new Promise(resolve => {
        finish = () => resolve(server(record.path, record.init, record.body));
      }) : undefined;
      await choose("task-select", "task-a");
      assert.equal(isBusy(), "true");
      finish(); await tick();
      assert.equal(isBusy(), "false");
      await emit("logout-button", "click");
      assert.equal(isBusy(), "false");
    ''')


def test_busy_persists_across_mutation_followup_and_clears_on_error(document):
    run_dom(document, r'''
      await login(); await choose("context-select", "ctx-a");
      let finish;
      intercept = record => record.path === "/api/audit" ? new Promise(resolve => {
        finish = () => resolve(server(record.path, record.init, record.body));
      }) : undefined;
      await emit("create-task", "click");
      assert.equal(mutations().length, 1);
      assert.equal($("workbench").getAttribute("aria-busy"), "true");
      finish(); await tick();
      assert.equal($("workbench").getAttribute("aria-busy"), "false");
      intercept = record => record.path.endsWith("/run")
        ? response({error: {code: "provider_failed", message: "合成测试拒绝"}}, 409) : undefined;
      await emit("run-task", "click");
      assert.equal($("error-message").hidden, false);
      assert.equal($("workbench").getAttribute("aria-busy"), "false");
    ''')


def test_cancel_independent_of_inflight_run_and_late_lower_version(document):
    run_dom(document, r'''
      await login(); await choose("task-select", "task-a");
      let finishRun;
      const original = clone(state.tasks[0]);
      intercept = record => record.path.endsWith("/run") ? new Promise(resolve => {finishRun = resolve;}) : undefined;
      await emit("run-task", "click");
      assert.equal($("run-task").disabled, true);
      assert.equal($("workbench").getAttribute("aria-busy"), "true");
      assert.equal($("cancel-task").disabled, false);
      await emit("cancel-task", "click");
      assert.equal(state.tasks[0].status, "cancelled");
      assert.equal($("workbench").getAttribute("aria-busy"), "true");
      finishRun(response(original)); await tick();
      assert.equal($("workbench").getAttribute("aria-busy"), "false");
      assert.match($("task-summary").textContent, /已取消/);
      assert.equal($("run-task").disabled, true);
      assert.equal(state.recommendations.length, 0);
    ''')


@pytest.mark.parametrize("action", ["logout", "reload", "pagehide"])
def test_session_epoch_aborts_and_ignores_late_responses(document, action):
    run_dom(document, r'''
      await login(); await choose("task-select", "task-a");
      let finish;
      const late = clone(state.tasks[0]);
      late.payload.error = "LATE_RESPONSE_MUST_NOT_RENDER"; late.version += 100;
      intercept = record => record.path.endsWith("/run") ? new Promise(resolve => {finish = resolve;}) : undefined;
      await emit("run-task", "click");
      const request = requests.find(r => r.path.endsWith("/run"));
      if ("ACTION" === "pagehide") windowListeners.pagehide();
      else await emit("ACTION" === "logout" ? "logout-button" : "reload-button", "click");
      assert.equal(request.init.signal.aborted, true);
      assert.equal($("workbench").getAttribute("aria-busy"), "false");
      finish(response(late)); await tick();
      assert.equal($("workbench").getAttribute("aria-busy"), "false");
      assert(!$("task-summary").textContent.includes("LATE_RESPONSE_MUST_NOT_RENDER"));
      if ("ACTION" !== "reload") {
        assert.equal($("workbench").hidden, true);
        assert.equal($("context-select").value, "");
        assert.equal($("token").value, "");
        assert.equal($("review-note").value, "");
      } else assert.equal($("workbench").hidden, false);
    '''.replace("ACTION", action))


def test_running_recovery_uses_backend_lease_guard_and_surfaces_busy(document):
    run_dom(document, r'''
      state.tasks[0] = task("task-a", "ctx-a", "blocked", {unknown_calls: 0, error: "provider_failed"});
      // Browser would call this expired, but only the server's lease decision counts.
      serverNow = Date.now() - 600000;
      state.tasks[1] = task("task-b", "ctx-b", "running", {unknown_calls: 1,
        lease_id: "lease-active", lease_until_ms: Date.now() - 300000});
      await login(); await choose("task-select", "task-a");
      assert.equal($("run-task").disabled, false);
      await choose("task-select", "task-b");
      assert.equal($("run-task").disabled, false);
      assert.match($("run-task").textContent, /后端核验租约/);
      assert.equal(mutations().length, 0);
      await emit("run-task", "click");
      assert.equal(mutations().length, 1);
      assert.deepEqual(mutations()[0].body, {});
      assert.match($("error-message").textContent, /busy/);
      assert.equal($("error-message").hidden, false);
      assert.equal(state.tasks[1].status, "running");
      assert.equal($("workbench").getAttribute("aria-busy"), "false");
      assert.equal(state.recommendations.length, 0);
      await tick(); assert.equal(mutations().length, 1);
    ''')


@pytest.mark.parametrize("lease", ["expired", "none", "browser_future"])
def test_stranded_running_unknown_can_explicitly_normalize_without_model_retry(document, lease):
    run_dom(document, r'''
      const until = "LEASE" === "browser_future" ? Date.now() + 300000 : Date.now() - 300000;
      if ("LEASE" === "browser_future") serverNow = Date.now() + 600000;
      state.tasks[0] = task("task-a", "ctx-a", "running", {calls: 1, unknown_calls: 1,
        lease_id: "LEASE" === "none" ? null : "lease-lost-process", lease_until_ms: until});
      await login(); await choose("task-select", "task-a");
      assert.equal($("run-task").disabled, false);
      assert.equal($("reconcile-form").hidden, false);
      assert.equal(mutations().length, 0);
      await emit("run-task", "click");
      assert.equal(mutations().length, 1);
      assert(mutations()[0].path.endsWith("/run"));
      assert.deepEqual(mutations()[0].body, {});
      assert.equal(state.tasks[0].status, "blocked");
      assert.equal(state.tasks[0].payload.calls, 1);
      assert.equal(state.tasks[0].payload.unknown_calls, 1);
      assert.match($("task-summary").textContent, /model_unknown/);
      assert.equal($("run-task").disabled, true);
      assert.equal($("reconcile-form").hidden, false);
      assert.equal($("reconcile-submit").disabled, true);
      assert.equal($("request-execution").disabled, true);
      assert.equal($("apply-execution").disabled, true);
      assert.equal(state.recommendations.length, 0);
      await emit("reload-button", "click"); await tick();
      assert.equal(mutations().length, 1);
    '''.replace("LEASE", lease))


@pytest.mark.parametrize("initial_status", ["blocked", "running", "cancelled"])
def test_admin_unknown_reconcile_available_across_recovery_states(document, initial_status):
    run_dom(document, r'''
      state.tasks[0] = task("task-a", "ctx-a", "INITIAL", {calls: 1, unknown_calls: 1,
        error: "model_unknown", lease_id: "lease-expired", lease_until_ms: serverNow - 1});
      await login(); await choose("task-select", "task-a");
      assert.equal($("reconcile-form").hidden, false);
      assert.equal($("reconcile-submit").disabled, true);
      assert.equal(mutations().length, 0);
      await fill("reconcile-call", "call-lost-process");
      await fill("reconcile-note", "已核查进程退出后的无结果记录");
      await check("reconcile-confirm"); await emit("reconcile-form", "submit");
      assert.equal(mutations().length, 1);
      assert(mutations()[0].path.endsWith("/reconcile"));
      assert.equal(mutations()[0].body.no_result, true);
      assert.equal(mutations()[0].body.tokens, null);
      assert.equal(state.tasks[0].payload.unknown_calls, 0);
      assert.equal(state.tasks[0].status, "INITIAL" === "cancelled" ? "cancelled" : "created");
      assert.equal($("reconcile-form").hidden, true);
      assert.equal($("run-task").disabled, "INITIAL" === "cancelled");
      assert.equal(state.recommendations.length, 0);
      assert.equal(state.executions.length, 0);
      await tick(); assert.equal(mutations().length, 1);
    '''.replace("INITIAL", initial_status))


def test_cancelled_task_retains_admin_reconcile_for_unknown_call(document):
    run_dom(document, r'''
      state.tasks[0] = task("task-a", "ctx-a", "running", {unknown_calls: 1, calls: 1,
        lease_id: "lease-expired", lease_until_ms: serverNow - 1});
      await login(); await choose("task-select", "task-a");
      await emit("cancel-task", "click");
      assert.equal(state.tasks[0].status, "cancelled");
      assert.equal($("run-task").disabled, true);
      assert.equal($("reconcile-form").hidden, false);
      await fill("reconcile-call", "call-cancelled"); await fill("reconcile-note", "取消后核查未知用量");
      await check("reconcile-confirm"); await emit("reconcile-form", "submit");
      assert.equal(mutations().length, 2);
      assert.equal(state.tasks[0].status, "cancelled");
      assert.equal(state.tasks[0].payload.unknown_calls, 0);
      assert.equal($("run-task").disabled, true);
      assert.equal($("reconcile-form").hidden, true);
      assert.equal(state.recommendations.length, 0);
    ''')


def test_reconcile_busy_guard_and_admin_scope_remain_decisive(document):
    run_dom(document, r'''
      state.tasks[0] = task("task-a", "ctx-a", "running", {unknown_calls: 1,
        lease_id: "lease-active", lease_until_ms: serverNow + 600000});
      await login(); await choose("task-select", "task-a");
      await fill("reconcile-call", "call-still-running"); await fill("reconcile-note", "明确请求后端核查");
      await check("reconcile-confirm"); await emit("reconcile-form", "submit");
      assert.match($("error-message").textContent, /busy/);
      assert.equal(state.tasks[0].payload.unknown_calls, 1);
      assert.equal(mutations().length, 1);
      await tick(); assert.equal(mutations().length, 1);
      await emit("logout-button", "click"); role = "operator";
      await login(); await choose("task-select", "task-a");
      assert.equal($("reconcile-form").hidden, true);
      await check("reconcile-confirm"); await emit("reconcile-form", "submit");
      assert.equal(mutations().length, 1);
      assert.equal($("request-execution").disabled, true);
    ''')


def test_permission_change_clears_identity_and_rendered_data(document):
    run_dom(document, r'''
      await login(); await choose("task-select", "task-a");
      intercept = record => record.path === "/api/me" ? response({code: "forbidden"}, 403) : undefined;
      await emit("reload-button", "click");
      assert.equal($("workbench").hidden, true);
      assert.equal($("context-select").value, "");
      assert.equal($("token").value, "");
      assert.match($("message").textContent, /权限已变化/);
      assert.equal($("run-task").disabled, true);
    ''')


def test_audit_and_unknown_usage_are_readable_and_admin_reconcile_explicit(document):
    run_dom(document, r'''
      state.tasks[0] = task("task-a", "ctx-a", "blocked", {tokens: null, unknown_calls: 1, error: "model_unknown"});
      state.audit = [{kind: "task", object_id: "task-a", action: "tool_completed",
        created_at_ms: Date.now(), detail: {tool: "predict_bc", tokens: null, note: "仅合成审计"}}];
      await login(); await choose("task-select", "task-a");
      assert.match($("audit-list").textContent, /仅合成审计/);
      assert.match($("task-summary").textContent, /未知（不是 0）/);
      assert.equal($("reconcile-form").hidden, false);
      assert.equal($("reconcile-submit").disabled, true);
      await fill("reconcile-call", "call-original"); await fill("reconcile-note", "依据本地回执确认无可用结果");
      await check("reconcile-confirm"); await emit("reconcile-form", "submit");
      const body = mutations()[0].body;
      assert.equal(body.tokens, null); assert.equal(body.no_result, true);
      assert.equal(body.call_id, "call-original"); assert(body.note);
      assert.equal(mutations().length, 1);
    ''')
