#!/usr/bin/env node
/** Actual installed Chrome CDP + parameter API/DOM; real NumPy BC, TEST-ONLY tool selector.
 * Safe default: --help/import/self-test never launches processes, network or file writes.
 * Explicit run with Node 24: node scripts/browser_smoke.mjs --run-chrome --artifacts-parent /private/tmp
 * A fresh owned directory OUTSIDE the repository holds the temporary app/profile. Only a
 * sanitized summary remains there after cleanup; no screenshots, response dumps or tokens.
 * Credentials cross only the private child stdout pipe and Chrome memory/Authorization.
 */
import assert from 'node:assert/strict';
import {spawn, execFile} from 'node:child_process';
import {constants} from 'node:fs';
import {access, lstat, mkdtemp, open, readFile, readdir, realpath, rm, writeFile} from 'node:fs/promises';
import {get as httpGet} from 'node:http';
import {dirname, isAbsolute, join, relative, resolve} from 'node:path';
import {createInterface} from 'node:readline';
import {fileURLToPath} from 'node:url';
import {promisify} from 'node:util';

export const PROJECT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
export const SELECTORS = Object.freeze({
  token: '#token', login: '#login-button', logout: '#logout-button', workspace: '#workbench',
  context: '#context-select', contextParameters: '#context-parameters', model: '#model-summary',
  create: '#create-task', run: '#run-task', task: '#task-select', cancel: '#cancel-task',
  evidence: '#task-evidence', candidate: '#task-candidate', recommendation: '#recommendation-select',
  recommendationCandidate: '#recommendation-candidate', draft: '#model-draft',
  decision: '#review-decision', note: '#review-note', review: '#review-submit',
  executionConfirm: '#execution-confirm', request: '#request-execution', execution: '#execution-select',
  applyConfirm: '#apply-confirm', apply: '#apply-execution', readback: '#readback-execution',
  observe: '#observe-recommendation', observation: '#observation-summary', refresh: '#reload-button',
});
const CHECKS = [
  'isolated_installed_chrome_and_TEST_server', 'anonymous_and_invalid_login',
  'login_real_identity_memory_only_credentials', 'actual_BC_and_original_context_values',
  'task_create_is_not_run', 'explicit_TEST_run_signed_evidence_and_frozen_candidate',
  'raw_quantized_target_values_equal_backend', 'separate_hash_bound_review_without_execution',
  'explicit_execution_request_without_apply', 'synthetic_apply_is_not_readback',
  'real_device_readback_matches_target', 'mature_observation_with_explicit_synthetic_clock',
  'real_lifecycle_audit_and_literal_text_safety', 'logout_rejects_late_real_API_response',
  'reload_forgets_identity', 'no_visible_raw_JSON_dump_or_persistent_browser_credentials',
  'security_headers_and_no_external_page_requests', 'owned_process_cleanup',
  'no_plaintext_tokens_in_owned_files', 'owned_runtime_directory_cleanup',
];
export const USAGE = `Parameter TEST-ONLY browser smoke (Node 24, installed macOS Chrome).
Usage:
  node scripts/browser_smoke.mjs --help
  node scripts/browser_smoke.mjs --self-test
  node scripts/browser_smoke.mjs --run-chrome --artifacts-parent /absolute/existing/outside/repo
      [--chrome /absolute/Chrome] [--python /absolute/python]
No browser starts without --run-chrome. No downloads, user profiles, ambient credentials,
server URL, production data, screenshots or raw API dumps. All lifecycle actions are UI clicks.
Opt-in trains actual NumPy BC on original fixtures; tool selection uses SyntheticProvider TEST ONLY.
Maturity uses an injected synthetic clock advanced after verified real readback, NOT 60s wall time
or physical evidence. Production ActionService is unchanged; no clock-control HTTP endpoint.
A new owned directory is created under artifacts-parent; only summary.json is retained.
`;

export function parseArgs(args) {
  if (args.length === 0 || (args.length === 1 && args[0] === '--help')) return {mode: 'help'};
  if (args.length === 1 && args[0] === '--self-test') return {mode: 'self-test'};
  const options = {mode: 'run', chrome: DEFAULT_CHROME, python: join(PROJECT, '.venv/bin/python')};
  const seen = new Set();
  for (let i = 0; i < args.length; i++) {
    const flag = args[i];
    assert(!seen.has(flag), 'Duplicate argument'); seen.add(flag);
    if (flag === '--run-chrome') { options.authorized = true; continue; }
    assert(['--artifacts-parent', '--chrome', '--python'].includes(flag), 'Unknown argument');
    const value = args[++i];
    assert(typeof value === 'string' && isAbsolute(value) && !/[\x00-\x1f\x7f]/.test(value),
      'An explicit absolute path is required');
    options[{'--artifacts-parent': 'parent', '--chrome': 'chrome', '--python': 'python'}[flag]] = value;
  }
  assert(options.authorized && options.parent, '--run-chrome AND --artifacts-parent are required');
  assert(!isWithin(PROJECT, resolve(options.parent)), 'Artifacts must be outside the repository');
  return options;
}
export function isWithin(parent, child) {
  const path = relative(parent, child);
  return path === '' || (!path.startsWith('../') && path !== '..' && !isAbsolute(path));
}
// Bootstrap-only wrapper, never installed as an application route or production middleware.
// Inert tests execute only these definitions with fake ASGI messages, never bootstrap().
export const PY_SERVER = String.raw`
import asyncio, json, secrets, socket, sys, threading, time
from pathlib import Path

class SyntheticClock:
    def __init__(self, initial_ms, advance_ms=60_001):
        if type(initial_ms) is not int or initial_ms < 0:
            raise ValueError('invalid synthetic clock origin')
        if type(advance_ms) is not int or not 60_000 <= advance_ms <= 120_000:
            raise ValueError('invalid synthetic clock interval')
        self.now_ms, self.advance_ms = initial_ms, advance_ms
        self.advances = 0
        self.seen = set()

    def __call__(self):
        return self.now_ms

    def verified_readback(self, execution):
        payload = execution.get('payload', {})
        readback = payload.get('readback', {})
        identity = execution.get('object_id')
        if (execution.get('kind') != 'execution' or execution.get('status') != 'executed'
                or not isinstance(identity, str) or identity in self.seen
                or readback.get('execution_id') != identity or readback.get('synthetic') is not True
                or type(readback.get('readback_at_ms')) is not int
                or readback['readback_at_ms'] != self.now_ms
                or not isinstance(readback.get('readback_hash'), str)
                or len(readback['readback_hash']) != 64
                or not payload.get('target') or readback.get('current') != payload['target']):
            return
        self.seen.add(identity)
        self.now_ms += self.advance_ms
        self.advances += 1

class SmokeHarness:
    def __init__(self, app, token, clock, gate, release):
        self.app, self.token, self.clock = app, token, clock
        self.gate, self.release = gate, release

    async def __call__(self, scope, receive, reply):
        http = scope['type'] == 'http'
        authorized = http and scope.get('headers', []).count(
            (b'authorization', ('Bearer ' + self.token).encode('ascii'))) == 1
        hold = (authorized and scope['method'] == 'GET'
                and scope['path'] == '/api/contexts' and self.gate['armed'])
        readback = (authorized and scope['method'] == 'POST'
                    and scope['path'].startswith('/api/executions/')
                    and scope['path'].endswith('/readback'))
        if hold:
            self.gate['armed'] = False
        status, body = None, bytearray()
        async def send_response(message):
            nonlocal status
            if message['type'] == 'http.response.start':
                status = message['status']
                if hold and status == 200:
                    self.gate['held'] = True
                    await asyncio.to_thread(self.release.wait, 15)
                    self.gate['held'] = False
                    self.gate['released'] += 1
            if readback and status == 200 and message['type'] == 'http.response.body':
                body.extend(message.get('body', b''))
                if len(body) > 2_000_000:
                    raise ValueError('readback response bound')
            # Always forward the actual production response, without rewriting its contents.
            await reply(message)
            if (readback and status == 200 and message['type'] == 'http.response.body'
                    and not message.get('more_body', False)):
                # This is synthetic clock time, never a wall-clock or physical observation claim.
                # Advance only AFTER successful real readback was sent, not on apply or a click.
                self.clock.verified_readback(json.loads(body))
        await self.app(scope, receive, send_response)

def bootstrap():
    import uvicorn
    from parameter_agent.api import create_app
    from parameter_agent.bc import train_model
    from parameter_agent.domain import digest
    from parameter_agent.fixtures import demo_contexts, generate_demonstrations
    from parameter_agent.provider import SyntheticProvider

    channel = sys.stdout
    sys.stdout = sys.stderr
    write_lock = threading.Lock()
    def send(value):
        with write_lock:
            channel.write(json.dumps(value, ensure_ascii=True) + '\n')
            channel.flush()
    try:
        data_root = Path(sys.argv[1])
        if data_root.exists():
            raise ValueError('bootstrap requires a new owned data directory')
        clock = SyntheticClock(time.time_ns() // 1_000_000)
        original = demo_contexts(clock())
        token = secrets.token_urlsafe(32)
        app = create_app(data_root, SyntheticProvider(), clock=clock)
        actor = app.state.store.create_user('browser-admin', token, 'admin',
                    sorted({context.plant_id for context in original.values()}))
        for case, context in original.items():
            app.state.service.import_context(actor, context, 'browser-import-' + case)
        # Actual default training, not stubbed predictions, hand-built weights or seed search.
        training = train_model(generate_demonstrations())
        plant_id = original['short_shot'].plant_id
        # Same model-hash/owner/plant evaluation binding used by the shipped offline train CLI.
        metrics_id = 'metrics-' + digest({'owner': actor.user_id, 'plant_id': plant_id,
                                          'model_hash': training['model']['model_hash']})
        with app.state.store.transaction() as connection:
            app.state.store.require(actor, plant_id, roles=('admin', 'reviewer'), conn=connection)
            app.state.store.create_object(connection, kind='model_metrics', object_id=metrics_id,
                principal=actor, plant_id=plant_id, status='evaluated',
                payload={'model_hash': training['model']['model_hash'], 'metrics': training['metrics']})
            app.state.store.audit(connection, actor, kind='model_metrics', object_id=metrics_id,
                action='synthetic_bc_evaluated', detail={'model_hash': training['model']['model_hash']})
        model = app.state.service.register_model(actor, training['model'],
                    plant_id, 'browser-register-model')
        # No task, recommendation, approval, execution, readback or observation is seeded.
        gate = {'armed': False, 'held': False, 'released': 0}
        release = threading.Event()
        server = uvicorn.Server(uvicorn.Config(SmokeHarness(app, token, clock, gate, release),
            host='127.0.0.1', log_level='critical', access_log=False, log_config=None,
            server_header=False, proxy_headers=False))
        def control():
            # Private inherited stdin pipe, never an HTTP control endpoint or environment token.
            for line in sys.stdin:
                msg = {}
                try:
                    msg = json.loads(line)
                    command = msg['command']
                    if command == 'hold_contexts':
                        release.clear(); gate['armed'] = True; value = {'armed': True}
                    elif command == 'release_contexts':
                        release.set(); value = {'released': True}
                    elif command == 'state':
                        value = {**gate, 'clock_advances': clock.advances,
                                 'synthetic_now_ms': clock(), 'advance_ms': clock.advance_ms}
                    elif command == 'shutdown':
                        release.set(); server.should_exit = True; value = {'stopping': True}
                    else:
                        raise ValueError('unsupported control')
                    send({'id': msg['id'], 'value': value})
                except Exception:
                    send({'id': msg.get('id'), 'error': 'fixture_control_failed'})
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', 0)); sock.listen(128)
        threading.Thread(target=control, daemon=True).start()
        send({'type': 'ready', 'port': sock.getsockname()[1], 'token': token,
              'user_id': actor.user_id, 'provider': app.state.provider.fingerprint,
              'contexts': {key: value.context_id for key, value in original.items()},
              'model_hash': training['model']['model_hash'], 'model_id': model['object_id'],
              'training': training['metrics']['training'],
              'clock': {'label': 'synthetic clock', 'initial_ms': clock(),
                        'advance_ms': clock.advance_ms, 'wall_wait_performed': False}})
        server.run(sockets=[sock])
    except Exception as exc:
        send({'type': 'failed', 'code': getattr(exc, 'code', type(exc).__name__)})
        raise SystemExit(1) from None

if __name__ == '__main__':
    bootstrap()
`;

class CDP {
  constructor(socket, fault) {
    this.socket = socket; this.next = 0; this.pending = new Map(); this.listeners = new Set();
    socket.addEventListener('message', event => {
      let value; try { value = JSON.parse(event.data); } catch { this.close(); return; }
      const pending = this.pending.get(value.id);
      if (pending) {
        clearTimeout(pending.timer); this.pending.delete(value.id);
        if (value.error) pending.reject(new Error(`CDP ${pending.method} failed`));
        else pending.resolve(value.result);
      } else for (const listener of this.listeners) Promise.resolve(listener(value)).catch(() => fault('CDP observer failed'));
    });
    socket.addEventListener('close', () => this.close());
    socket.addEventListener('error', () => this.close());
  }
  send(method, params = {}, sessionId) {
    assert(this.socket.readyState === WebSocket.OPEN, 'CDP unavailable');
    return new Promise((resolvePromise, reject) => {
      const id = ++this.next;
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error(`CDP ${method} timeout`)); }, 12000);
      this.pending.set(id, {resolve: resolvePromise, reject, timer, method});
      this.socket.send(JSON.stringify({id, method, params, ...(sessionId ? {sessionId} : {})}));
    });
  }
  close() {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer); pending.reject(new Error('CDP closed'));
    }
    this.pending.clear();
    if (this.socket.readyState === WebSocket.OPEN) this.socket.close();
  }
}
const delay = ms => new Promise(resolvePromise => setTimeout(resolvePromise, ms));
async function deadline(promise, ms, label) {
  let timer;
  try { return await Promise.race([promise, new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(label)), ms);
  })]); } finally { clearTimeout(timer); }
}

export class Smoke {
  constructor(options) {
    this.options = options; this.started = Date.now(); this.children = []; this.secrets = [];
    this.results = []; this.responses = []; this.requests = new Map(); this.controls = new Map();
    this.faults = []; this.external = new Set(); this.sequence = 0; this.stopping = false;
    this.owned = new Map(); this.knownProcesses = new Map();
  }
  safe(value) {
    let text = String(value);
    for (const secret of this.secrets) text = text.split(secret).join('[REDACTED]');
    // AssertionError may append entire compared objects; retain only its human headline.
    return text.split(/\r?\n/, 1)[0].slice(0, 800);
  }
  record(name, status, detail) {
    this.results.push({name, status, ...(detail ? {detail: this.safe(detail)} : {})});
  }
  async check(name, work) {
    await work();
    await this.snapshotOwnedProcesses();
    this.record(name, 'pass');
  }
  async until(work, label, timeout = 12000) {
    const end = Math.min(Date.now() + timeout, this.started + 240000);
    while (!this.stopping && Date.now() < end) {
      const result = await work(); if (result) return result;
      await delay(40);
    }
    throw new Error(`Timed out: ${label}`);
  }
  environment() {
    return {PATH: '/usr/bin:/bin:/usr/sbin:/sbin', HOME: this.runtime, TMPDIR: this.runtime,
      LANG: 'en_US.UTF-8', PYTHONPATH: join(PROJECT, 'src'),
      PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1'};
  }
  launch(executable, args, stdio) {
    assert(!this.stopping, 'No child launches during cleanup');
    const child = spawn(executable, args, {cwd: this.runtime, env: this.environment(), detached: true, stdio});
    this.children.push(child);
    child.on('error', () => { child.launchFailed = true; });
    child.stdin?.on('error', () => { this.faults.push('Owned child control pipe closed'); });
    child.stderr?.resume();
    return child;
  }
  async remember(path) {
    const info = await lstat(path);
    assert(info.isDirectory() && !info.isSymbolicLink(), 'Owned path must be a directory');
    this.owned.set(path, {dev: info.dev, ino: info.ino});
  }
  async ownedDirectory(path) {
    const info = await lstat(path); const owned = this.owned.get(path);
    assert(owned && info.isDirectory() && !info.isSymbolicLink() && owned.dev === info.dev && owned.ino === info.ino,
      'Owned directory identity changed; refusing recursive cleanup');
  }
  async start() {
    assert(process.platform === 'darwin', 'Installed macOS Chrome is required');
    assert(process.versions.node.split('.')[0] === '24' && typeof WebSocket === 'function',
      'Explicit installed Node 24 with built-in WebSocket is required');
    const parent = await realpath(this.options.parent);
    assert((await lstat(parent)).isDirectory() && !isWithin(await realpath(PROJECT), parent),
      'Artifacts parent must be an existing directory outside this repository');
    await Promise.all([this.options.chrome, this.options.python].map(path => access(path, constants.X_OK)));
    this.root = await mkdtemp(join(parent, 'parameter-browser-'));
    await this.remember(this.root);
    this.runtime = await mkdtemp(join(this.root, 'owned-runtime-'));
    await this.remember(this.runtime);
    this.server = this.launch(this.options.python, ['-u', '-c', PY_SERVER, join(this.runtime, 'app')], ['pipe', 'pipe', 'ignore']);
    this.bootstrap = await deadline(new Promise((resolvePromise, reject) => {
      createInterface({input: this.server.stdout}).on('line', line => {
        let value; try { value = JSON.parse(line); } catch { reject(new Error('Invalid private setup protocol')); return; }
        if (value.type === 'ready') {
          this.secrets.push(value.token); resolvePromise(value); return;
        }
        if (value.type === 'failed') { reject(new Error('TEST app bootstrap failed (diagnostics withheld)')); return; }
        const pending = this.controls.get(value.id);
        if (pending) { clearTimeout(pending.timer); this.controls.delete(value.id);
          if (value.error) pending.reject(new Error('TEST control failed')); else pending.resolve(value.value); }
      });
      this.server.once('exit', () => reject(new Error('TEST server exited before readiness')));
      this.server.once('error', () => reject(new Error('Project Python launch failed')));
    }), 120000, 'TEST app and actual NumPy BC training startup deadline');
    assert(Number.isInteger(this.bootstrap.port) && this.bootstrap.port > 0 && this.bootstrap.port <= 65535,
      'Invalid owned server port');
    await this.snapshotOwnedProcesses();
    this.origin = `http://127.0.0.1:${this.bootstrap.port}`;
    const profile = join(this.runtime, 'chrome-profile');
    this.chrome = this.launch(this.options.chrome, [
      '--headless=new', '--remote-debugging-address=127.0.0.1', '--remote-debugging-port=0',
      `--user-data-dir=${profile}`, '--no-first-run', '--no-default-browser-check',
      '--disable-background-networking', '--disable-component-update', '--disable-default-apps', '--disable-sync',
      '--disable-breakpad', '--disable-crash-reporter', '--disable-domain-reliability',
      '--safebrowsing-disable-auto-update', '--disable-client-side-phishing-detection',
      '--metrics-recording-only', '--password-store=basic', '--use-mock-keychain',
      '--no-proxy-server', '--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1',
      '--disable-features=AutofillServerCommunication,OptimizationHints,MediaRouter,Translate,PasswordLeakDetection',
      '--window-size=1440,1100', 'about:blank',
    ], ['ignore', 'ignore', 'ignore']);
    const active = await this.until(async () => {
      assert(this.chrome.exitCode === null && !this.chrome.launchFailed, 'Owned Chrome exited before readiness');
      try {
        const lines = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).trim().split('\n');
        return /^\d+$/.test(lines[0]) && Number(lines[0]) <= 65535 &&
          /^\/devtools\/browser\/[A-Za-z0-9-]+$/.test(lines[1] || '') ? lines : null;
      } catch { return null; }
    }, 'isolated DevToolsActivePort', 20000);
    const socket = new WebSocket(`ws://127.0.0.1:${active[0]}${active[1]}`);
    await deadline(new Promise((resolvePromise, reject) => {
      socket.addEventListener('open', resolvePromise, {once: true});
      socket.addEventListener('error', () => reject(new Error('Owned Chrome CDP failed')), {once: true});
    }), 10000, 'CDP connection deadline');
    this.cdp = new CDP(socket, fault => this.faults.push(fault));
    const version = await this.cdp.send('Browser.getVersion');
    assert(/Chrome\//.test(version.product), 'Actual Chrome product was not reported');
    const target = await this.cdp.send('Target.createTarget', {url: 'about:blank'});
    this.sessionId = (await this.cdp.send('Target.attachToTarget', {targetId: target.targetId, flatten: true})).sessionId;
    this.cdp.listeners.add(event => this.observe(event));
    await this.cmd('Page.enable'); await this.cmd('Runtime.enable'); await this.cmd('Network.enable');
    const healthResponse = await this.until(async () => {
      assert(this.server.exitCode === null && !this.server.launchFailed, 'Owned TEST server exited');
      return new Promise(resolvePromise => {
        // Bare numeric-loopback HTTP: no proxy env, redirects, retries or caller URL.
        const request = httpGet({hostname: '127.0.0.1', port: this.bootstrap.port, path: '/health',
          agent: false, timeout: 1000}, response => {
          let raw = ''; response.setEncoding('utf8');
          response.on('data', data => { raw += data; if (raw.length > 4096) request.destroy(); });
          response.on('end', () => {
            try {
              const body = JSON.parse(raw);
              resolvePromise(response.statusCode === 200 && body.synthetic_only === true &&
                typeof body.provider === 'string' && body.provider.startsWith('synthetic-test-') ? body : null);
            } catch { resolvePromise(null); }
          });
          response.on('error', () => resolvePromise(null));
        });
        request.on('timeout', () => request.destroy());
        request.on('error', () => resolvePromise(null));
      });
    }, 'TEST server HTTP readiness');
    assert.equal(healthResponse.provider, this.bootstrap.provider, 'Unexpected TEST provider identity');
    await this.cmd('Page.navigate', {url: this.origin});
    await this.until(() => this.evaluate(`document.readyState === 'complete' && !!document.querySelector('#login-form')`), 'Parameter login DOM');
    const health = await this.probe('/health');
    assert(health.status === 200 && health.body.synthetic_only === true &&
      health.body.provider === this.bootstrap.provider, 'Smoke server is not explicitly TEST-ONLY');
    await this.snapshotOwnedProcesses();
    this.record('isolated_installed_chrome_and_TEST_server', 'pass', version.product);
  }
  cmd(method, params = {}) { return this.cdp.send(method, params, this.sessionId); }
  async evaluate(expression) {
    const value = await this.cmd('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
    assert(!value.exceptionDetails, 'Browser evaluation failed (payload withheld)');
    return value.result?.value;
  }
  async observe(event) {
    if (event.sessionId !== this.sessionId) return;
    const p = event.params;
    if (event.method === 'Runtime.exceptionThrown') this.faults.push('Uncaught browser exception');
    if (event.method === 'Runtime.consoleAPICalled' && this.secrets.some(secret => JSON.stringify(p.args).includes(secret)))
      this.faults.push('Credential in browser console');
    if (event.method === 'Page.javascriptDialogOpening') {
      this.faults.push('Unexpected browser dialog'); await this.cmd('Page.handleJavaScriptDialog', {accept: false});
    }
    if (event.method === 'Network.requestWillBeSent') {
      const url = new URL(p.request.url);
      if (['http:', 'https:', 'ws:', 'wss:'].includes(url.protocol) && url.origin !== this.origin)
        this.external.add(url.origin);
      if (this.secrets.some(secret => p.request.url.includes(secret) || (p.request.postData || '').includes(secret)))
        this.faults.push('Credential in URL/request body');
      // Authorization headers are intentionally NEVER retained.
      this.requests.set(p.requestId, {path: url.pathname, method: p.request.method, rawRequest: p.request.postData});
    }
    if (event.method === 'Network.responseReceived') {
      const request = this.requests.get(p.requestId);
      if (request) this.responses.push({...request, requestId: p.requestId, status: p.response.status,
        headers: p.response.headers, cached: Boolean(p.response.fromDiskCache || p.response.fromServiceWorker), done: false});
    }
    if (event.method === 'Network.loadingFailed') {
      // A real 401 handler/logout may abort the fetch before Chrome exposes its body.
      // Preserve only an already observed status; never invent a successful response.
      const entry = this.responses.find(row => row.requestId === p.requestId && !row.done);
      if (entry) { entry.body = null; entry.done = true; entry.aborted = true; }
    }
    if (event.method === 'Network.loadingFinished') {
      const entry = this.responses.find(row => row.requestId === p.requestId && !row.done);
      if (!entry) return;
      if (entry.path.startsWith('/api/') || entry.path === '/health') {
        try {
          const result = await this.cmd('Network.getResponseBody', {requestId: p.requestId});
          const raw = result.base64Encoded ? Buffer.from(result.body, 'base64').toString('utf8') : result.body;
          assert(raw.length < 2_000_000, 'API response exceeds smoke bound');
          if (this.secrets.some(secret => raw.includes(secret))) this.faults.push('Credential in API response');
          entry.body = JSON.parse(raw);
        } catch { entry.body = null; }
      }
      entry.done = true;
    }
  }
  async click(selector) {
    const point = await this.evaluate(`(() => {
      const e = document.querySelector(${JSON.stringify(selector)});
      if (!e || e.disabled || e.closest('[hidden]')) return null;
      e.scrollIntoView({block:'center',behavior:'instant'}); const r = e.getBoundingClientRect();
      return r.width && r.height ? {x:r.x+r.width/2,y:r.y+r.height/2} : null;
    })()`);
    assert(point, `Not a visible enabled click target: ${selector}`);
    await this.cmd('Input.dispatchMouseEvent', {type: 'mousePressed', button: 'left', clickCount: 1, ...point});
    await this.cmd('Input.dispatchMouseEvent', {type: 'mouseReleased', button: 'left', clickCount: 1, ...point});
  }
  async fill(selector, value) {
    assert(await this.evaluate(`(() => {const e = document.querySelector(${JSON.stringify(selector)});
      if (!e || e.disabled || e.closest('[hidden]')) return false;
      e.scrollIntoView({block:'center',behavior:'instant'}); e.focus(); e.select(); return true;})()`), `Missing input: ${selector}`);
    await this.cmd('Input.insertText', {text: String(value)});
  }
  async choose(selector, value) {
    const state = await this.evaluate(`(() => {const e = document.querySelector(${JSON.stringify(selector)});
      if (!e) return 'missing select';
      if (e.disabled) return 'disabled select';
      if (![...e.options].some(o => o.value === ${JSON.stringify(value)})) return 'missing option';
      e.value = ${JSON.stringify(value)}; e.dispatchEvent(new Event('change',{bubbles:true})); return 'selected';
    })()`);
    assert.equal(state, 'selected', `Cannot choose ${selector}: ${state}`);
  }
  async idle() {
    await this.until(() => this.evaluate(`document.querySelector('#workbench')?.getAttribute('aria-busy') === 'false'`),
      'UI operation and refresh idle');
  }
  async responseAfter(mark, path, method = 'POST') {
    return this.until(() => this.responses.slice(mark).find(row => row.path === path && row.method === method && row.done),
      `${method} ${path}`, 20000);
  }
  async submit(selector, path, method = 'POST') {
    const mark = this.responses.length;
    await this.click(selector);
    const response = await this.responseAfter(mark, path, method); await this.idle();
    assert(response.status === 200 && response.body, `UI operation failed: ${method} ${path} HTTP ${response.status}`);
    return response;
  }
  async probe(path, authenticated = false) {
    assert(path.startsWith('/') && !path.startsWith('//'), 'Only same-origin probes');
    const headers = authenticated ? {Authorization: `Bearer ${this.bootstrap.token}`} : {};
    return this.evaluate(`(async () => {const r = await fetch(${JSON.stringify(path)}, {
      headers:${JSON.stringify(headers)}, credentials:'omit', cache:'no-store', mode:'same-origin', redirect:'error'});
      return {status:r.status, body:await r.json()};})()`);
  }
  async control(command) {
    const id = ++this.sequence;
    return new Promise((resolvePromise, reject) => {
      const timer = setTimeout(() => { this.controls.delete(id); reject(new Error('TEST control deadline')); }, 5000);
      this.controls.set(id, {resolve: resolvePromise, reject, timer});
      this.server.stdin.write(JSON.stringify({id, command}) + '\n');
    });
  }
  count(path) { return [...this.requests.values()].filter(row => row.method === 'POST' && row.path === path).length; }
  async disabled(selector) { return this.evaluate(`document.querySelector(${JSON.stringify(selector)})?.disabled === true`); }
  async text(selector) { return this.evaluate(`document.querySelector(${JSON.stringify(selector)})?.textContent || ''`); }
  async unchecked(selector) { return this.evaluate(`document.querySelector(${JSON.stringify(selector)})?.checked === false`); }
  requestBody(response, keys) {
    const body = JSON.parse(response.rawRequest);
    assert.deepEqual(Object.keys(body).sort(), keys.sort(), 'UI request contains missing/extra/automatic flags');
    return body;
  }
  async login() {
    await this.fill(SELECTORS.token, this.bootstrap.token);
    const result = (await this.submit(SELECTORS.login, '/api/me', 'GET')).body;
    assert.equal(result.user_id, this.bootstrap.user_id);
    assert.equal(result.role, 'admin');
    await this.until(() => this.evaluate(`!document.querySelector('#workbench').hidden &&
      document.querySelector('#login-form').hidden`), 'authenticated workbench visible');
    assert((await this.text('#provider-status')).includes('synthetic-test-'), 'TEST disclosure missing');
    assert(await this.evaluate(`document.querySelector('#token').value === ''`), 'Login retained token input');
    await this.storageCheck();
  }
  async storageCheck() {
    const storage = await this.evaluate(`(async () => ({local:localStorage.length, session:sessionStorage.length,
      cookies:document.cookie, databases:await indexedDB.databases(), caches:await caches.keys(),
      registrations:(await navigator.serviceWorker.getRegistrations()).length,
      tokenInput:document.querySelector('#token').value}))()`);
    assert(storage.local === 0 && storage.session === 0 && storage.cookies === '' &&
      storage.databases.length === 0 && storage.caches.length === 0 && storage.registrations === 0 && storage.tokenInput === '',
    'Browser persisted credentials or retained token input');
    const browserCookies = await this.cmd('Network.getCookies', {urls: [this.origin]});
    assert(browserCookies.cookies.length === 0, 'Browser cookie storage is not empty');
    const dom = await this.evaluate('document.documentElement.outerHTML');
    assert(!this.secrets.some(secret => dom.includes(secret)), 'Credential rendered into DOM');
  }
  async noRawDump() {
    assert(await this.evaluate(`![...document.querySelectorAll('#workbench pre')].some(e =>
      !e.closest('[hidden]') && !e.closest('details:not([open])') && e.getClientRects().length > 0)`),
    'A raw JSON dump is visible instead of the structured UI');
  }
  async rows(selector) {
    return this.evaluate(`([...document.querySelectorAll(${JSON.stringify(selector + ' tbody tr')})].map(row =>
      [...row.querySelectorAll('td')].map(cell => cell.textContent)))`);
  }
  async candidateCheck(selector, candidate, context) {
    const names = ['temperature', 'speed', 'pressure', 'cooling'];
    const labels = ['温度', '注射速度', '保压压力', '冷却时间'];
    const directions = {increase:'增加', decrease:'减少', hold:'不变'};
    const expected = names.map((name, i) => [labels[i], context.current[name], candidate.raw_delta[name],
      candidate.delta[name], candidate.target[name], directions[candidate.directions[name]], context.units[name]]);
    assert.deepEqual(await this.rows(selector), expected, 'Displayed raw delta/quantized delta/target/unit differs from backend');
    assert(candidate.eligible === true && candidate.issues.length === 0, 'Candidate is not hard-gate eligible');
    assert((await this.text(selector)).includes(candidate.candidate_hash), 'Candidate binding hash not displayed');
    await this.noRawDump();
  }
  async cleared() {
    const result = await this.evaluate(`({hidden:document.querySelector('#workbench').hidden,
      loginHidden:document.querySelector('#login-form').hidden, user:document.querySelector('#session-status').textContent,
      values:[...document.querySelectorAll('input,textarea')].map(e => e.type === 'checkbox' ? e.checked : e.value),
      ids:['context-select','task-select','recommendation-select','execution-select'].map(id =>
        [...document.getElementById(id).options].filter(o=>o.value).length),
      panels:['context-summary','task-summary','task-candidate','recommendation-summary',
        'execution-detail','observation-summary','audit-list'].map(id=>document.getElementById(id).textContent)})`);
    assert(result.hidden && !result.loginHidden && result.user === '未登录', 'Logged-out identity revived');
    assert(result.values.every(value => value === '' || value === false) && result.ids.every(count => count === 0),
      'Logout retained sensitive inputs/objects');
    for (const id of [this.task?.object_id, this.recommendation?.object_id, this.execution?.object_id].filter(Boolean))
      assert(!result.panels.some(panel => panel.includes(id)), 'Logout retained an old object in DOM');
    await this.storageCheck();
  }
  async run() {
    await this.start();
    await this.check('anonymous_and_invalid_login', async () => {
      assert.equal((await this.probe('/api/me')).status, 401);
      await this.fill(SELECTORS.token, 'browser-wrong-token');
      const mark = this.responses.length; await this.click(SELECTORS.login);
      assert.equal((await this.responseAfter(mark, '/api/me', 'GET')).status, 401); await this.idle();
      assert((await this.text('#message')).includes('身份'), 'Invalid token error not shown');
      await this.cleared();
    });
    await this.check('login_real_identity_memory_only_credentials', () => this.login());
    await this.check('actual_BC_and_original_context_values', async () => {
      await this.choose(SELECTORS.context, this.bootstrap.contexts.short_shot);
      const contexts = await this.probe('/api/contexts', true);
      this.context = contexts.body.items.find(row => row.object_id === this.bootstrap.contexts.short_shot).payload;
      const names = ['temperature', 'speed', 'pressure', 'cooling'];
      const labels = ['温度', '注射速度', '保压压力', '冷却时间'];
      assert.deepEqual(await this.rows(SELECTORS.contextParameters), names.map((name, i) =>
        [labels[i], this.context.current[name], this.context.units[name]]), 'Context parameter strings/units differ from API');
      const models = (await this.probe('/api/models', true)).body.items;
      const model = models.find(row => row.object_id === this.bootstrap.model_id);
      assert(model && model.payload.model_hash === this.bootstrap.model_hash, 'Real trained model not exposed in UI API');
      assert(!('model' in model.payload) && !('weights' in model.payload), 'API exposes model weights');
      const training = this.bootstrap.training;
      assert(training.final_loss < training.initial_loss && training.epochs === 240 && training.test_used_for_tuning === false,
        'Bootstrap did not train the default real BC model');
      assert.deepEqual(model.payload.metrics.training, training, 'HTTP model metrics differ from actual training');
      assert(model.payload.metrics.synthetic === true && model.payload.metrics.test, 'Bound held-out synthetic evaluation is missing');
      const modelText = await this.text(SELECTORS.model);
      assert(modelText.includes(this.bootstrap.model_hash), 'Trained model identity not displayed');
      for (const value of [training.initial_loss, training.final_loss, training.epochs])
        assert(modelText.includes(String(value)), 'Actual trained metrics are not displayed');
      assert.equal(this.count('/api/tasks'), 0, 'Login/selection auto-created a task');
    });
    await this.check('task_create_is_not_run', async () => {
      const response = await this.submit(SELECTORS.create, '/api/tasks');
      const posted = this.requestBody(response, ['action_id', 'context_id']);
      assert.equal(posted.context_id, this.context.context_id);
      this.task = response.body;
      assert(this.task.status === 'created' && this.task.payload.calls === 0 && this.task.payload.tokens === 0 &&
        this.task.payload.candidate === null, 'Task creation auto-ran inference');
      assert.equal(this.count(`/api/tasks/${this.task.object_id}/run`), 0);
      assert.equal((await this.probe('/api/recommendations', true)).body.items.length, 0);
    });
    await this.check('explicit_TEST_run_signed_evidence_and_frozen_candidate', async () => {
      const response = await this.submit(SELECTORS.run, `/api/tasks/${this.task.object_id}/run`);
      this.requestBody(response, []); this.task = response.body;
      assert(this.task.status === 'submitted' && this.task.payload.calls > 0 && this.task.payload.tokens > 0,
        'Real default BC/TEST run did not submit; no fixture/model fallback is allowed');
      this.recommendation = (await this.probe(`/api/recommendations/${this.task.payload.recommendation_id}`, true)).body;
      const rec = this.recommendation, candidate = rec.payload.candidate;
      assert(rec.status === 'pending_review' && !rec.payload.approval && !rec.payload.execution, 'Submission auto-approved/executed');
      assert.deepEqual(candidate, this.task.payload.candidate, 'Submitted candidate changed after checking');
      for (const tool of ['get_process_context', 'retrieve_experience', 'predict_bc']) {
        const evidence = rec.payload.evidence.find(item => item.tool === tool);
        assert(evidence && evidence.task_id === this.task.object_id && evidence.context_hash === rec.payload.context_hash &&
          /^[a-f0-9]{64}$/.test(evidence.content_hash), 'Issued evidence missing or unbound');
        assert((await this.text(SELECTORS.evidence)).includes(evidence.evidence_id), 'Real tool evidence not displayed');
      }
      assert.equal(await this.text(SELECTORS.draft), rec.payload.untrusted_model_draft, 'Model text differs from untrusted backend draft');
      assert(!(await this.text(SELECTORS.recommendationCandidate)).includes(rec.payload.untrusted_model_draft),
        'Untrusted model draft merged into authoritative candidate');
      assert(await this.disabled(SELECTORS.review) && await this.disabled(SELECTORS.request));
      assert(await this.evaluate(`document.querySelector('#review-decision').value === '' && document.querySelector('#review-note').value === ''`),
        'Human approval was prefilled');
      assert.equal((await this.probe('/api/executions', true)).body.items.length, 0);
    });
    await this.check('raw_quantized_target_values_equal_backend', async () => {
      await this.candidateCheck(SELECTORS.candidate, this.task.payload.candidate, this.context);
      await this.candidateCheck(SELECTORS.recommendationCandidate, this.recommendation.payload.candidate, this.context);
    });
    const recPath = `/api/recommendations/${this.recommendation.object_id}`;
    const literal = '<img id="parameter-smoke-literal" src="/smoke-never-image" onerror="window.__smokeXSS=1">';
    const note = `TEST ONLY synthetic human review; not physical validation or production approval. ${literal}`;
    await this.check('separate_hash_bound_review_without_execution', async () => {
      await this.choose(SELECTORS.decision, 'approved');
      assert(await this.disabled(SELECTORS.review), 'Review note is not required');
      await this.fill(SELECTORS.note, note);
      const response = await this.submit(SELECTORS.review, `${recPath}/review`);
      const posted = this.requestBody(response, ['action_id', 'recommendation_hash', 'decision', 'note']);
      assert(posted.recommendation_hash === this.recommendation.content_hash && posted.decision === 'approved' && posted.note === note);
      this.recommendation = response.body;
      assert(this.recommendation.status === 'approved' && this.recommendation.payload.approval.note === note &&
        !this.recommendation.payload.execution, 'Approval executed a device action');
      assert(await this.unchecked(SELECTORS.executionConfirm) && await this.disabled(SELECTORS.request));
      assert.equal(this.count(`${recPath}/request-execution`), 0);
    });
    await this.check('explicit_execution_request_without_apply', async () => {
      await this.click(SELECTORS.executionConfirm);
      const response = await this.submit(SELECTORS.request, `${recPath}/request-execution`);
      const posted = this.requestBody(response, ['action_id', 'recommendation_hash', 'confirmed']);
      assert(posted.recommendation_hash === this.recommendation.content_hash && posted.confirmed === true);
      this.execution = response.body;
      assert(this.execution.status === 'pending' && !this.execution.payload.readback && !this.execution.payload.device_receipt_hash);
      assert.deepEqual(this.execution.payload.target, this.recommendation.payload.candidate.target);
      assert.deepEqual(this.execution.payload.before, this.context.current);
      assert(await this.unchecked(SELECTORS.applyConfirm) && await this.disabled(SELECTORS.apply));
      assert(await this.disabled(SELECTORS.readback) && await this.disabled(SELECTORS.observe));
      assert.equal(this.count(`/api/executions/${this.execution.object_id}/apply`), 0);
    });
    const executionPath = `/api/executions/${this.execution.object_id}`;
    await this.check('synthetic_apply_is_not_readback', async () => {
      await this.click(SELECTORS.applyConfirm);
      const response = await this.submit(SELECTORS.apply, `${executionPath}/apply`);
      this.requestBody(response, ['action_id']); this.execution = response.body;
      assert(this.execution.status === 'applied' && this.execution.payload.adapter_acknowledged === true &&
        /^[a-f0-9]{64}$/.test(this.execution.payload.device_receipt_hash) && !this.execution.payload.readback);
      assert(await this.disabled(SELECTORS.apply) && await this.disabled(SELECTORS.observe));
      assert.equal((await this.control('state')).clock_advances, 0, 'Clock advanced before verified readback');
      assert.equal(this.count(`${executionPath}/readback`), 0);
      assert.equal(this.count(`${recPath}/observe`), 0);
    });
    await this.check('real_device_readback_matches_target', async () => {
      const response = await this.submit(SELECTORS.readback, `${executionPath}/readback`);
      this.requestBody(response, ['action_id']); this.execution = response.body;
      const p = this.execution.payload, readback = p.readback;
      assert(this.execution.status === 'executed' && readback.synthetic === true && readback.device_revision === p.expected_device_revision + 1);
      assert.deepEqual(readback.current, p.target, 'Actual local device readback differs from frozen target');
      assert.equal(readback.device_receipt_hash, p.device_receipt_hash);
      assert((await this.text('#execution-detail')).includes(readback.readback_hash), 'Verified readback evidence not rendered');
      assert.equal(this.count(`${recPath}/observe`), 0, 'Readback auto-observed');
      const state = await this.control('state');
      assert(state.clock_advances === 1 && state.synthetic_now_ms - readback.readback_at_ms === state.advance_ms);
    });
    await this.check('mature_observation_with_explicit_synthetic_clock', async () => {
      const response = await this.submit(SELECTORS.observe, `${recPath}/observe`);
      this.requestBody(response, ['action_id']); this.recommendation = response.body;
      const o = this.recommendation.payload.observation;
      assert(this.recommendation.status === 'observed' && o.synthetic === true && o.age_after_readback_ms >= 60000);
      assert(o.causal_effect_proven === false && o.baseline_is_observed_production === false && o.independent_samples === true);
      assert(o.before.sample_count === 120 && o.after.sample_count === 120 && o.before_seed === 27183 && o.after_seed === 27184,
        'Observe changed the fixed server samples/seed');
      assert.equal(o.difference_after_minus_before, o.after.defect_rate - o.before.defect_rate);
      assert(['synthetic_observed_improvement', 'no_clear_improvement', 'worsened'].includes(o.conclusion));
      assert(o.difference_ci95.length === 2 && o.difference_ci95.every(Number.isFinite));
      const text = await this.text(SELECTORS.observation);
      for (const value of [o.before.defect_count, o.before.sample_count, o.after.defect_count,
        o.after.sample_count, o.difference_after_minus_before, ...o.difference_ci95])
        assert(text.includes(String(value)), 'Observation values differ from actual API');
      assert(text.includes('不是已观测生产基线') && text.includes('不证明现场因果'), 'Synthetic observation caveat missing');
      assert.deepEqual(this.recommendation.payload.candidate, this.task.payload.candidate, 'Observation changed frozen parameters');
      this.observationEvidence = {clock: 'synthetic clock', synthetic_advance_ms: this.bootstrap.clock.advance_ms,
        wall_wait_performed: false, physical_evidence: false, conclusion: o.conclusion};
    });
    await this.check('real_lifecycle_audit_and_literal_text_safety', async () => {
      const audit = (await this.probe('/api/audit', true)).body.items;
      const actions = audit.map(item => item.action), text = await this.text('#audit-list');
      for (const action of ['review', 'request_execution', 'synthetic_apply', 'readback_verified', 'synthetic_observation'])
        assert(actions.includes(action) && text.includes(action), 'Actual lifecycle audit event missing');
      assert((await this.text('#recommendation-summary')).includes(literal), 'Human note not displayed literally');
      assert(await this.evaluate(`!window.__smokeXSS && !document.querySelector('#parameter-smoke-literal') && !document.querySelector('#workbench img')`),
        'Untrusted text became executable HTML');
      for (const action of ['review', 'request-execution', 'observe']) assert.equal(this.count(`${recPath}/${action}`), 1);
      for (const action of ['apply', 'readback']) assert.equal(this.count(`${executionPath}/${action}`), 1);
      const ids = [...this.requests.values()].filter(row => row.method === 'POST' && row.rawRequest)
        .map(row => JSON.parse(row.rawRequest).action_id).filter(Boolean);
      assert(ids.every(id => /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(id)) && new Set(ids).size === ids.length,
        'UI action IDs are not independent cryptographic UUIDs');
    });
    await this.check('no_visible_raw_JSON_dump_or_persistent_browser_credentials', async () => {
      await this.noRawDump(); await this.storageCheck();
    });
    await this.check('logout_rejects_late_real_API_response', async () => {
      await this.control('hold_contexts'); await this.click(SELECTORS.refresh);
      await this.until(async () => (await this.control('state')).held, 'real contexts response held');
      await this.click(SELECTORS.logout); await this.cleared();
      await this.control('release_contexts');
      await this.until(async () => (await this.control('state')).released > 0, 'real response released');
      await delay(250); await this.cleared();
    });
    await this.check('reload_forgets_identity', async () => {
      await this.login(); await this.cmd('Page.reload', {ignoreCache: true});
      await this.until(() => this.evaluate(`document.readyState === 'complete' && document.querySelector('#workbench')?.hidden === true`), 'reload clears identity');
      await this.cleared();
      assert.equal((await this.probe('/api/me')).status, 401, 'Refresh retained ambient browser authentication');
    });
    await this.check('security_headers_and_no_external_page_requests', async () => {
      const responses = this.responses.filter(row => (row.path.startsWith('/api/') || row.path.startsWith('/static/') || row.path === '/') && row.done);
      assert(responses.length > 20 && responses.every(row => !row.cached), 'Insufficient uncached actual API traffic');
      for (const response of responses) {
        const headers = Object.fromEntries(Object.entries(response.headers).map(([key, value]) => [key.toLowerCase(), value]));
        assert(headers['cache-control'] === 'no-store' && headers['x-content-type-options'] === 'nosniff' &&
          headers['referrer-policy'] === 'no-referrer' && headers['x-frame-options'] === 'DENY' &&
          headers['content-security-policy']?.includes("default-src 'self'"), 'Security headers missing');
      }
      assert.equal(this.external.size, 0, 'Page attempted outside-origin traffic');
      assert.equal(this.faults.length, 0, this.faults.join('; '));
    });
  }
  async snapshotOwnedProcesses() {
    if (!this.children.length) return [];
    const table = await this.processTable();
    // Remember start times while parents are alive so known descendants stay owned
    // even if a parent crashes/reparents them before final cleanup. Never match names.
    const pids = new Set(this.children.filter(child => child.pid && !child.launchFailed &&
      child.exitCode === null && child.signalCode === null).map(child => child.pid));
    for (const row of table) if (this.knownProcesses.get(row.pid)?.started === row.started) pids.add(row.pid);
    let changed = true;
    while (changed) {
      changed = false;
      for (const row of table) if (pids.has(row.parent) && !pids.has(row.pid)) { pids.add(row.pid); changed = true; }
    }
    for (const row of table) if (pids.has(row.pid)) this.knownProcesses.set(row.pid, row);
    return table.filter(row => pids.has(row.pid));
  }
  async processTable() {
    const {stdout} = await promisify(execFile)('/bin/ps', ['-axo', 'pid=,ppid=,lstart='], {maxBuffer: 2 * 1024 * 1024, timeout: 3000});
    return stdout.trim().split('\n').map(line => {
      const [pid, parent, ...started] = line.trim().split(/\s+/);
      return {pid: Number(pid), parent: Number(parent), started: started.join(' ')};
    });
  }
  killGroup(child, signal) {
    if (!child?.pid || child.launchFailed || child.exitCode !== null || child.signalCode !== null) return;
    try { process.kill(-child.pid, signal); } catch (error) { if (error.code !== 'ESRCH') throw error; }
  }
  async cleanup() {
    this.stopping = true;
    let descendants = [], enumerationFailed = false;
    try {
      if (this.children.length) {
        const owned = await this.snapshotOwnedProcesses();
        descendants = owned.filter(row => !this.children.some(child => child.pid === row.pid)).reverse();
      }
    } catch { enumerationFailed = true; }
    this.cdp?.close();
    for (const pending of this.controls.values()) { clearTimeout(pending.timer); pending.reject(new Error('Smoke cleanup')); }
    this.controls.clear();
    try {
      for (const child of this.children) this.killGroup(child, 'SIGTERM');
      await Promise.race([Promise.all(this.children.map(child => child.exitCode !== null || child.signalCode !== null || child.launchFailed
        ? Promise.resolve() : new Promise(resolvePromise => child.once('exit', resolvePromise)))), delay(2000)]);
      for (const child of this.children) this.killGroup(child, 'SIGKILL');
      if (descendants.length) {
        const current = await this.processTable();
        for (const owned of descendants) if (current.some(row => row.pid === owned.pid && row.started === owned.started)) {
          try { process.kill(owned.pid, 'SIGKILL'); } catch (error) { if (error.code !== 'ESRCH') throw error; }
        }
      }
      await delay(200);
      assert(this.children.every(child => child.exitCode !== null || child.signalCode !== null || child.launchFailed), 'Owned direct child still running');
      if (descendants.length) {
        const remaining = await this.processTable();
        assert(!descendants.some(owned => remaining.some(row => row.pid === owned.pid && row.started === owned.started)), 'Owned descendant still running');
      }
      assert(!enumerationFailed, 'Owned process enumeration failed; only direct groups could be reaped');
      this.record('owned_process_cleanup', 'pass');
    } catch (error) {
      for (const child of this.children) { try { this.killGroup(child, 'SIGKILL'); } catch {} }
      this.record('owned_process_cleanup', 'fail', error.message);
    }
    if (this.runtime) {
      try {
        await this.ownedDirectory(this.root); await this.ownedDirectory(this.runtime);
        const needles = this.secrets.flatMap(secret => [Buffer.from(secret), Buffer.from(secret, 'utf16le')]);
        let files = 0, total = 0;
        const scan = async path => {
          for (const entry of await readdir(path, {withFileTypes: true})) {
            if (entry.isSymbolicLink()) continue; // Never follow Chrome lock links or an outside target.
            const file = join(path, entry.name);
            if (entry.isDirectory()) await scan(file);
            else if (entry.isFile()) {
              files++; assert(files <= 20000, 'Owned file scan count bound exceeded');
              const handle = await open(file, constants.O_RDONLY | constants.O_NOFOLLOW);
              try {
                const stat = await handle.stat(); assert(stat.isFile(), 'Nonregular owned file');
                total += stat.size; assert(total <= 512 * 1024 * 1024, 'Owned file scan size bound exceeded');
                let previous = Buffer.alloc(0), chunk = Buffer.alloc(65536);
                const overlap = Math.max(1, ...needles.map(needle => needle.length)) - 1;
                for (;;) {
                  const {bytesRead} = await handle.read(chunk, 0, chunk.length, null); if (!bytesRead) break;
                  const combined = Buffer.concat([previous, chunk.subarray(0, bytesRead)]);
                  assert(!needles.some(needle => combined.includes(needle)), 'Plaintext token persisted in an owned file (content withheld)');
                  previous = Buffer.from(combined.subarray(Math.max(0, combined.length - overlap)));
                }
              } finally { await handle.close(); }
            }
          }
        };
        await scan(this.runtime);
        this.record('no_plaintext_tokens_in_owned_files', 'pass', `${files} owned files scanned after shutdown; content not retained`);
      } catch (error) { this.record('no_plaintext_tokens_in_owned_files', 'fail', error.message); }
      try {
        await this.ownedDirectory(this.root); await this.ownedDirectory(this.runtime);
        await rm(this.runtime, {recursive: true, force: false, maxRetries: 3, retryDelay: 100});
        this.record('owned_runtime_directory_cleanup', 'pass');
      } catch (error) { this.record('owned_runtime_directory_cleanup', 'fail', error.message); }
    }
  }
  async report() {
    for (const name of CHECKS) if (!this.results.some(row => row.name === name)) this.record(name, 'not_tested', 'Prerequisite failed; no UI/API bypass');
    const counts = Object.fromEntries(['pass', 'fail', 'not_tested'].map(status => [status, this.results.filter(row => row.status === status).length]));
    const report = {status: counts.fail || counts.not_tested ? 'fail' : 'pass', mode: 'actual Chrome CDP + real parameter API + actual NumPy BC; SyntheticProvider TEST-ONLY',
      observation: this.observationEvidence || {clock: 'synthetic clock', wall_wait_performed: false, physical_evidence: false, status: 'not_tested'},
      elapsed_seconds: Number(((Date.now() - this.started) / 1000).toFixed(2)), counts, results: this.results,
      not_tested: ['actual LLM/runtime', 'wall-clock 60s maturity', 'production/physical action', 'multi-tab concurrency', 'other browsers', 'OS-level browser background traffic'],
      artifacts: this.results.some(row => row.name === 'owned_runtime_directory_cleanup' && row.status === 'pass')
        ? 'Only sanitized summary retained; owned app/profile removed; no screenshots/raw API or credential dumps.'
        : 'Runtime cleanup was not verified. Inspect the cleanup result; do not assume app/profile deletion.'};
    const output = this.safeReport(report);
    if (this.root) {
      await this.ownedDirectory(this.root);
      await writeFile(join(this.root, 'summary.json'), output, {flag: 'wx', mode: 0o600});
    }
    process.stdout.write(output);
    if (this.root) process.stdout.write(`Owned summary: ${this.safe(join(this.root, 'summary.json'))}\n`);
    return report.status === 'pass' ? 0 : 1;
  }
  safeReport(report) {
    let output = JSON.stringify(report, null, 2) + '\n';
    for (const secret of this.secrets) output = output.split(secret).join('[REDACTED]');
    return output;
  }
}

export function selfTest() {
  assert.equal(parseArgs([]).mode, 'help'); assert.equal(parseArgs(['--self-test']).mode, 'self-test');
  for (const args of [['--run-chrome'], ['--artifacts-parent', '/private/tmp'], ['--run-chrome', '--artifacts-parent', PROJECT],
    ['--run-chrome', '--artifacts-parent', '/private/tmp', '--model-url', 'http://127.0.0.1'],
    ['--run-chrome', '--artifacts-parent', '/private/tmp', '--chrome', 'relative'],
    ['--run-chrome', '--run-chrome', '--artifacts-parent', '/private/tmp']]) assert.throws(() => parseArgs(args));
  assert.equal(parseArgs(['--run-chrome', '--artifacts-parent', '/private/tmp']).authorized, true);
  const smoke = new Smoke({}); smoke.secrets.push('TEST-SECRET');
  assert(!smoke.safeReport({message: 'TEST-SECRET'}).includes('TEST-SECRET'));
  process.stdout.write('PASS inert argument, no-launch default, redaction and isolated-path self-tests\n');
}

async function main() {
  let options;
  try { options = parseArgs(process.argv.slice(2)); }
  catch { process.stderr.write(USAGE); process.exitCode = 2; return; }
  if (options.mode === 'help') { process.stdout.write(USAGE); return; }
  if (options.mode === 'self-test') { selfTest(); return; }
  const smoke = new Smoke(options);
  let interrupt;
  const interrupted = new Promise((_, reject) => { interrupt = reject; });
  const stop = reason => {
    if (smoke.stopping) return;
    smoke.stopping = true;
    smoke.cdp?.close();
    // The cleanup phase enumerates descendants BEFORE terminating their live parents.
    interrupt(new Error(reason));
  };
  const timer = setTimeout(() => stop('250s absolute smoke deadline'), 250000);
  const signals = ['SIGINT', 'SIGTERM', 'SIGHUP'];
  const handlers = signals.map(signal => { const handler = () => stop(`Interrupted by ${signal}`); process.on(signal, handler); return handler; });
  const work = smoke.run();
  try { await Promise.race([work, interrupted]); }
  catch (error) { smoke.record('browser_smoke', 'fail', error.message); }
  finally {
    smoke.stopping = true;
    // Wait for bounded startup/DOM operations to settle before deleting any directory.
    // In particular, a signal cannot race mkdtemp and leave an untracked owned profile.
    await work.catch(() => {});
    clearTimeout(timer);
    await smoke.cleanup();
    try { process.exitCode = await smoke.report(); }
    catch { process.stderr.write('Parameter smoke summary could not be retained safely.\n'); process.exitCode = 1; }
    signals.forEach((signal, i) => process.removeListener(signal, handlers[i]));
  }
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
