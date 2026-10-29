(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const PARAMETERS = ["temperature", "speed", "pressure", "cooling"];
  const UNITS = { temperature: "degC", speed: "mm/s", pressure: "MPa", cooling: "s" };
  const LABELS = {
    temperature: "温度", speed: "注射速度", pressure: "保压压力", cooling: "冷却时间",
    context_id: "上下文 ID", plant_id: "工厂", machine_id: "设备", material_id: "材料",
    mold_id: "模具", process_version: "工艺版本", revision: "修订", observed_at_ms: "观测时间",
    defect: "缺陷", severity: "严重程度", current: "当前参数", units: "单位",
    material_viscosity: "材料黏度指标（合成无量纲）", ambient_c: "环境温度（degC）",
    sample_count: "样本数 / 分母", measurement_ok: "测量合格", synthetic: "合成数据",
    status: "状态", version: "对象版本", content_hash: "内容哈希", context_hash: "上下文哈希",
    model_hash: "模型哈希", model_version: "模型版本", model_content_hash: "模型对象哈希",
    context_version: "上下文版本", constraint_version: "约束版本", experience_version: "经验版本",
    task_id: "任务 ID", recommendation_id: "推荐 ID", execution_id: "执行 ID",
    source_snapshot: "绑定的来源快照", candidate_hash: "候选哈希", recommendation_hash: "推荐哈希",
    evidence_id: "证据 ID", tool: "签发工具", query: "检索条件", items: "条目", text: "内容",
    provenance: "来源", confidence: "动作置信度（非效果）", combination: "参数组合",
    raw_delta: "BC 原始增量", delta: "量化增量", target: "目标参数", before: "应用前参数 / 模拟基线",
    after: "应用后模拟样本", directions: "方向", eligible: "硬约束资格", issues: "检查问题",
    calls: "模型调用数 / 记录", tools: "工具调用数", tokens: "已知计费用量（不是未知调用总用量）",
    unknown_calls: "仍有未知 / 进行中调用数（用量不得当作 0）", provider_fingerprint: "模型提供方指纹",
    elapsed_s: "耗时（秒）", error: "停止 / 错误原因", pending_action: "待处理动作",
    awaiting_fields: "待补字段", lease_id: "租约 ID", lease_until_ms: "租约截至时间",
    prediction: "BC 预测", model: "模型", metrics: "合成评测", training_loss: "实际训练损失",
    train_loss: "训练损失", test: "测试集", calibration: "校准集", train: "训练集",
    split: "分组划分", splits: "分组划分", temperature_scaling: "校准温度",
    accuracy: "准确率", combination_accuracy: "参数组合准确率", delta_mae: "增量平均绝对误差",
    no_action: "无动作对照", simple_rule: "简单规则对照", group_id: "血缘分组",
    approval: "人工审核记录", review: "审核记录", decision: "决定", note: "人工说明",
    user_id: "操作者", reviewed_at_ms: "审核时间", approved_at_ms: "批准时间",
    request: "执行请求", request_id: "请求 ID", confirmed: "显式确认", owner: "所属用户",
    device_revision: "设备修订", expected_revision: "预期设备修订", applied_at_ms: "应用时间",
    readback_at_ms: "回读时间", readback_hash: "回读证据哈希", readback: "回读证据",
    device_receipt: "真实本地合成设备回执", receipt: "设备回执", observation: "成熟合成观察",
    observed_at: "观察时间", defect_count: "缺陷数 / 分子", defect_rate: "缺陷率",
    expected_defect_rate: "合成期望缺陷率", difference: "差值", delta_rate: "缺陷率差值",
    defect_rate_difference: "缺陷率差值", confidence_interval: "不确定性区间",
    interval: "不确定性区间", uncertainty: "二项采样不确定性", conclusion: "合成观察结论",
    causal_effect_proven: "是否证明现场因果效果", call_id: "模型调用 ID", no_result: "无可用结果",
    created_at_ms: "记录时间", updated_at_ms: "更新时间", action: "审计动作", detail: "审计详情",
    schema_version: "制品格式版本", feature_order: "特征顺序", seed: "服务端固定采样种子",
    sample_count_before: "比较前样本数", sample_count_after: "比较后样本数",
    difference_after_minus_before: "缺陷率差值（后 − 前，负值为下降）",
    difference_ci95: "差值 95% 不确定性区间", rate_ci95: "缺陷率 95% 区间",
    interval_method: "区间计算方法", age_after_readback_ms: "有效回读后时间（毫秒）",
    before_seed: "比较前服务端固定种子", after_seed: "比较后服务端固定种子",
    paired_simulated_comparison: "同一冻结上下文配对模拟", independent_samples: "两组样本独立采样",
    baseline_is_observed_production: "是否为已观测生产基线", notice: "说明",
    source: "绑定来源", source_hash: "来源哈希", request_hash: "请求哈希",
    expected_device_version: "预期设备对象版本", expected_device_revision: "预期设备修订",
    recommendation_version: "推荐版本", requested_at_ms: "请求时间", request_action_id: "请求动作 ID",
    training: "实际训练", epochs: "训练轮数", batch_size: "批大小", initial_loss: "初始损失",
    final_loss: "最终损失", normalization_source: "归一化拟合来源", test_used_for_tuning: "是否使用测试集调参",
    history: "训练损失记录", epoch: "轮次", loss: "损失", classification_loss: "组合分类损失",
    regression_loss: "增量回归损失", interpretation: "评测解释", expert_objective: "合成专家目标",
    bc: "行为克隆模仿指标", effect: "合成效果指标", groups: "血缘组数", records: "记录数",
    train_groups: "训练组", calibration_groups: "校准组", test_groups: "独立留出测试组",
    nll_before: "校准前 NLL", nll_after: "校准后 NLL",
    method: "方法", rows: "记录条数", confidence_threshold: "拒绝动作置信度阈值",
    records_hash: "分组记录哈希", normalized_delta_mae: "归一化增量平均绝对误差",
    evaluation_passes: "冻结后测试评估次数", accepted_combination_accuracy: "接受动作的组合准确率",
    no_action_combination_accuracy: "无动作对照组合准确率", coverage: "接受动作覆盖率",
    acceptance_rate: "动作接受率", expected_defect_rates: "合成期望缺陷率对照"
  };
  const WORDS = {
    created: "已创建，尚未运行", running: "运行中", awaiting_input: "等待人工补缺",
    blocked: "已阻断，需核查原因", submitted: "已提交推荐，未自动批准", no_recommendation: "未形成推荐",
    cancelled: "已取消", pending_review: "待人工审核", approved: "已批准，未自动执行", rejected: "已拒绝",
    execution_pending: "已请求合成执行", pending: "等待显式应用", applied: "已应用，尚未回读确认",
    unknown: "结果未知，禁止盲目重发", executed: "已回读确认", observed: "已有成熟合成观察",
    ok: "通过", abstain: "拒绝给出动作", short_shot: "短射", flash: "飞边", warpage: "翘曲", normal: "正常",
    increase: "增加", decrease: "减少", unchanged: "不变", up: "增加", down: "减少", hold: "不变",
    synthetic_observed_improvement: "合成样本观察到改善（不证明现场因果）",
    no_clear_improvement: "没有明确改善", worsened: "合成样本变差", synthetic: "仅合成",
    admin: "管理员", reviewer: "审核员", operator: "操作员",
    get_process_context: "过程上下文", retrieve_experience: "经验检索", predict_bc: "BC 行为克隆预测",
    draft_recommendation: "候选草拟", check_constraints: "硬约束核验",
    request_human_input: "人工补缺请求", submit_recommendation: "提交推荐",
    missing_material_viscosity: "缺少材料黏度指标", missing_ambient_c: "缺少环境温度",
    measurement_not_ok: "测量未通过", insufficient_samples: "样本不足", context_expired: "上下文已过期",
    context_from_future: "观测时间在未来", units_mismatch: "参数单位不匹配",
    temperature_cooling_coupling: "温度与冷却耦合约束未通过",
    speed_pressure_coupling: "速度与压力耦合约束未通过"
  };
  let token = "";
  let epoch = 0;
  let refreshSerial = 0;
  const controllers = new Set();
  const pendingReads = new Set();
  const busy = new Set();
  const emptyData = () => ({ contexts: [], models: [], tasks: [], recommendations: [], executions: [], audit: [] });
  let data = emptyData();
  let me = null;
  let stateReady = false;
  let contextId = "";
  let taskId = "";
  let recommendationId = "";
  let executionId = "";
  let reviewBinding = "";
  let executionBinding = "";
  let supplementBinding = "";

  const object = (kind, id) => data[kind].find((entry) => entry.object_id === id) || null;
  const payload = (entry) => entry && entry.payload && typeof entry.payload === "object" ? entry.payload : {};
  const status = (entry) => entry ? entry.status : "";
  const label = (key) => LABELS[key] || key;
  const word = (value) => WORDS[value] || value;
  const stamp = (value) => {
    if (typeof value !== "number" || !Number.isFinite(value)) return "未记录";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "无效时间" : date.toLocaleString("zh-CN", { hour12: false });
  };
  function scalar(value, key = "") {
    if (value === null || value === undefined) return key === "tokens" ? "未知（不是 0）" : "未提供";
    if (typeof value === "boolean") return value ? "是" : "否";
    if (key.endsWith("_at_ms") || key.endsWith("_until_ms")) return stamp(value);
    if (typeof value === "string") return String(WORDS[value] || LABELS[value] || value);
    if (typeof value === "number") return String(value);
    return "结构化记录";
  }
  function node(tag, text, className) {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = String(text);
    if (className) element.className = className;
    return element;
  }
  function put(id, ...children) {
    const element = $(id);
    element.replaceChildren(...children);
    element.classList.remove("empty");
  }
  function empty(id, text) {
    put(id, node("p", text, "muted"));
  }
  function pairs(entries) {
    const list = node("dl", undefined, "facts");
    entries.forEach(([key, value]) => {
      list.append(node("dt", label(key)), node("dd", scalar(value, key)));
    });
    return list;
  }
  // Render structured facts, never executable markup or a JSON blob.
  function readable(value, depth = 0, key = "") {
    if (value === null || typeof value !== "object") return node("span", scalar(value, key));
    if (depth > 7) return node("span", "深层记录已折叠，请按上层证据核查。", "muted");
    if (Array.isArray(value)) {
      if (!value.length) return node("span", "无条目", "muted");
      const list = node("ul", undefined, "fact-list");
      value.slice(0, 100).forEach((item) => {
        const row = node("li");
        row.append(readable(item, depth + 1, key));
        list.append(row);
      });
      if (value.length > 100) list.append(node("li", "仅展示前 100 条。"));
      return list;
    }
    const result = node("dl", undefined, "facts nested");
    Object.entries(value).filter(([name]) => !["weights", "layers", "normalizer"].includes(name)).slice(0, 100)
      .forEach(([name, item]) => {
        const term = node("dt", label(name));
        const detail = node("dd");
        const metricSummary = key === "metrics" && name === "calibration" && item && typeof item === "object"
          ? Object.fromEntries(Object.entries(item).map(([field, value]) => [field === "temperature" ? "temperature_scaling" : field, value]))
          : item;
        detail.append(readable(metricSummary, depth + 1, name));
        result.append(term, detail);
      });
    return result;
  }
  function table(headers, rows, caption) {
    const result = node("table");
    if (caption) result.append(node("caption", caption));
    const thead = node("thead");
    const tr = node("tr");
    headers.forEach((text) => { const th = node("th", text); th.scope = "col"; tr.append(th); });
    thead.append(tr);
    const tbody = node("tbody");
    rows.forEach((values) => {
      const row = node("tr");
      values.forEach((value) => {
        const cell = node("td");
        cell.append(value && typeof value === "object" && value.nodeType ? value : node("span", value));
        row.append(cell);
      });
      tbody.append(row);
    });
    result.append(thead, tbody);
    return result;
  }
  function options(id, items, selected, placeholder, description) {
    const select = $(id);
    const initial = node("option", placeholder);
    initial.value = "";
    const children = [initial];
    items.forEach((entry) => {
      const option = node("option", description(entry));
      option.value = entry.object_id;
      children.push(option);
    });
    select.replaceChildren(...children);
    select.value = items.some((entry) => entry.object_id === selected) ? selected : "";
  }
  function showMessage(message) { $("message").textContent = message; }
  function clearError() { $("error-message").textContent = ""; $("error-message").hidden = true; }
  function showError(error) {
    if (error.name === "AbortError") return;
    $("error-message").textContent = error.message || "操作失败；请重载核查服务端状态。";
    $("error-message").hidden = false;
  }
  // Includes response parsing/rendering, not just fetch; stale epochs cannot revive busy state.
  function updateBusy() {
    $("workbench").setAttribute("aria-busy", busy.size > 0 || controllers.size > 0 || pendingReads.size > 0 ? "true" : "false");
  }
  function invalidate() {
    epoch += 1;
    refreshSerial += 1;
    controllers.forEach((controller) => controller.abort());
    controllers.clear();
    pendingReads.clear();
    busy.clear();
    updateBusy();
  }
  function resetForms() {
    ["supplement-form", "review-form", "request-execution-form", "apply-form", "reconcile-form"].forEach((id) => $(id).reset());
    reviewBinding = "";
    executionBinding = "";
    supplementBinding = "";
  }
  function logout(message = "已退出，内存身份与页面数据已清空。") {
    invalidate();
    token = "";
    me = null;
    stateReady = false;
    data = emptyData();
    contextId = taskId = recommendationId = executionId = "";
    resetForms();
    $("token").value = "";
    $("login-form").hidden = false;
    $("session-controls").hidden = true;
    $("workbench").hidden = true;
    $("session-status").textContent = "未登录";
    $("provider-status").textContent = "尚未连接";
    $("login-button").disabled = false;
    clearError();
    render();
    showMessage(message);
  }
  async function api(path, body, expectedEpoch = epoch) {
    if (expectedEpoch !== epoch || !token) throw new DOMException("会话已变更", "AbortError");
    const controller = new AbortController();
    controllers.add(controller);
    updateBusy();
    try {
      const headers = { Accept: "application/json", Authorization: `Bearer ${token}` };
      const init = { method: body === undefined ? "GET" : "POST", headers,
        credentials: "omit", cache: "no-store", redirect: "error", signal: controller.signal };
      if (body !== undefined) {
        headers["Content-Type"] = "application/json";
        init.body = JSON.stringify(body);
      }
      const response = await fetch(path, init);
      if (expectedEpoch !== epoch) throw new DOMException("会话已变更", "AbortError");
      if (response.status === 401 || response.status === 403) {
        logout("身份无效或权限已变化，已中止请求并清空页面，请重新登录。");
        throw new DOMException("会话已结束", "AbortError");
      }
      let result;
      try { result = await response.json(); }
      catch { throw new Error(`服务端未返回有效 JSON（HTTP ${response.status}），请重载核查状态。`); }
      if (expectedEpoch !== epoch) throw new DOMException("会话已变更", "AbortError");
      if (!response.ok) {
        const detail = result.error || result.detail || result;
        const code = typeof detail === "object" && !Array.isArray(detail) ? detail.code : null;
        const message = typeof detail === "string" ? detail : detail.message;
        // No validation input/body echoes: they may contain sensitive values.
        throw new Error(`请求被拒绝（HTTP ${response.status}${code ? ` / ${code}` : ""}）：${typeof message === "string" ? message : "输入、权限或阶段不符合契约"}。未认定成功；请核查当前状态。`);
      }
      return result;
    } finally { controllers.delete(controller); updateBusy(); }
  }
  function accept(entry) {
    const plural = { context: "contexts", model: "models", task: "tasks", recommendation: "recommendations", execution: "executions" }[entry && entry.kind];
    if (!plural || !entry.object_id || !entry.payload) throw new Error("服务端对象形状不符合 metadata + payload 契约。");
    const old = object(plural, entry.object_id);
    if (old && old.version > entry.version) return;
    data[plural] = old ? data[plural].map((item) => item.object_id === entry.object_id ? entry : item) : [entry, ...data[plural]];
  }
  async function refresh(expectedEpoch) {
    const serial = ++refreshSerial;
    const names = ["contexts", "models", "tasks", "recommendations", "executions", "audit"];
    const values = await Promise.all(names.map((name) => api(`/api/${name}`, undefined, expectedEpoch)));
    if (expectedEpoch !== epoch || serial !== refreshSerial) return;
    names.forEach((name, index) => {
      if (!values[index] || !Array.isArray(values[index].items)) throw new Error(`${name} 列表缺少 items，不能确认状态。`);
    });
    names.forEach((name, index) => {
      data[name] = values[index].items.map((entry) => {
        if (name === "audit") return entry;
        const old = object(name, entry.object_id);
        return old && old.version > entry.version ? old : entry;
      });
    });
    if (!object("contexts", contextId)) contextId = "";
    if (!object("tasks", taskId)) taskId = "";
    if (!object("recommendations", recommendationId)) recommendationId = "";
    if (!object("executions", executionId)) executionId = "";
    stateReady = true;
    render();
  }
  function actionId() {
    if (!globalThis.crypto || typeof globalThis.crypto.randomUUID !== "function") {
      throw new Error("浏览器不支持安全 action_id，请使用受支持的本机浏览器。");
    }
    return crypto.randomUUID();
  }
  async function perform(key, operation, success) {
    if (!me || busy.has(key) || busy.has("reload")) return;
    const expectedEpoch = epoch;
    busy.add(key);
    clearError();
    showMessage("请求已发送，等待服务端确认；不会自动进入下一阶段。");
    controls();
    try {
      await operation(expectedEpoch);
      if (expectedEpoch !== epoch) return;
      render();
      await refresh(expectedEpoch);
      if (expectedEpoch === epoch) showMessage(success);
    } catch (error) {
      if (expectedEpoch === epoch) {
        stateReady = false;
        resetForms();
        showMessage("本次操作未确认完成；动作已锁定，请先重载服务端状态。不要盲目重试应用。");
        showError(error);
      }
    } finally {
      if (expectedEpoch === epoch) { busy.delete(key); controls(); }
    }
  }
  function unit(context, name) { return context && context.units ? (context.units[name] || "单位缺失") : `${UNITS[name]}（契约单位）`; }
  function parameters(values, context, caption) {
    return table(["参数", "值（精确十进制）", "单位"], PARAMETERS.map((name) => [label(name), scalar(values && values[name]), unit(context, name)]), caption);
  }
  function boundContext(candidate, evidence) {
    for (const item of evidence || []) {
      if (item.tool !== "get_process_context" || item.context_hash !== candidate.context_hash) continue;
      const p = item.payload || {};
      if (p.context) return p.context;
      if (p.current && p.units) return p;
      if (p.payload && p.payload.current) return p.payload;
    }
    const entry = data.contexts.find((item) => item.content_hash === candidate.context_hash);
    return entry ? payload(entry) : null;
  }
  function candidateView(id, candidate, evidence) {
    if (!candidate || !candidate.raw_delta) { empty(id, "尚无服务端冻结候选；模型文本不能替代候选。"); return; }
    const context = boundContext(candidate, evidence);
    const badge = node("p", candidate.eligible === true ? "硬约束核验通过（仍需独立人工审核）" : "未获硬约束执行资格", candidate.eligible === true ? "good" : "bad");
    const rows = PARAMETERS.map((name) => [label(name), scalar(context && context.current && context.current[name]),
      scalar(candidate.raw_delta[name]), scalar(candidate.delta && candidate.delta[name]),
      scalar(candidate.target && candidate.target[name]), scalar(candidate.directions && candidate.directions[name]), unit(context, name)]);
    const wrap = node("div", undefined, "table-wrap");
    wrap.append(table(["参数", "绑定原值", "原始 Δ", "量化 Δ", "目标", "方向", "实际单位"], rows, "增量不是百分比；精确值以服务端冻结结果为准"));
    const checks = node("section");
    checks.append(node("h4", "硬约束检查问题"), readable(candidate.issues || []));
    put(id, badge, wrap, checks, pairs([...["candidate_hash", "context_hash", "model_hash", "constraint_version"].map((key) => [key, candidate[key]])]));
  }
  function evidenceView(id, evidence) {
    if (!Array.isArray(evidence) || !evidence.length) { empty(id, "尚无服务端签发证据。"); return; }
    const cards = evidence.map((item) => {
      const card = node("details", undefined, "evidence-card");
      card.open = ["retrieve_experience", "predict_bc"].includes(item.tool);
      card.append(node("summary", `${word(item.tool)} · ${item.evidence_id || "未标识证据"}`),
        pairs(["task_id", "context_hash", "content_hash"].map((key) => [key, item[key]])), readable(item.payload));
      return card;
    });
    put(id, ...cards);
  }
  function renderContext() {
    const entry = object("contexts", contextId);
    options("context-select", data.contexts, contextId, "请选择上下文，不自动创建任务", (item) => `${item.object_id} · ${word(payload(item).defect)} · ${payload(item).machine_id}`);
    if (!entry) {
      empty("context-summary", "选择上下文后查看工艺信息。列表为空时请由本地 CLI 初始化原创合成输入。");
      put("context-parameters"); empty("model-summary", "请选择上下文以查看对应工厂的 BC 模型。"); return;
    }
    const p = payload(entry);
    put("context-summary", pairs([...["context_id", "plant_id", "machine_id", "material_id", "mold_id", "process_version", "revision", "observed_at_ms", "defect", "severity", "sample_count", "measurement_ok", "material_viscosity", "ambient_c", "synthetic"].map((key) => [key, p[key]]), ["version", entry.version], ["content_hash", entry.content_hash]]));
    put("context-parameters", parameters(p.current, p, "当前工艺四参数"));
    const models = data.models.filter((item) => item.plant_id === entry.plant_id);
    if (!models.length) { empty("model-summary", "本工厂没有本人可见的 BC 模型。推荐将阻断；请先通过本地 CLI 真实训练并注册，不自动回退。"); return; }
    put("model-summary", ...models.map((model) => {
      const section = node("section");
      const mp = payload(model);
      section.append(pairs([["model", model.object_id], ["plant_id", model.plant_id], ["status", model.status], ["version", model.version], ["content_hash", model.content_hash]]));
      const summary = Object.fromEntries(Object.entries(mp).filter(([key]) => !["model", "weights", "layers", "normalizer"].includes(key)));
      section.append(readable(summary));
      if (!mp.metrics || !Object.keys(mp.metrics).length) section.append(node("p", "未提供 metrics 评测摘要，不假设评测通过。", "muted"));
      return section;
    }));
  }
  function renderTask() {
    options("task-select", data.tasks, taskId, "请选择任务", (item) => `${item.object_id} · ${word(item.status)} · ${payload(item).context_id}`);
    $("task-count").textContent = `${data.tasks.length} 个任务（服务端限量）`;
    if (!data.tasks.length) empty("task-list", "暂无任务。");
    else put("task-list", table(["任务", "上下文", "状态", "操作"], data.tasks.map((item) => {
      const open = node("button", item.object_id === taskId ? "重新打开" : "打开 / 恢复", "secondary compact");
      open.type = "button"; open.dataset.taskId = item.object_id;
      open.disabled = busy.has("reload");
      open.addEventListener("click", () => selectTask(item.object_id));
      return [item.object_id, payload(item).context_id, word(item.status), open];
    }), "选择既有任务会重新绑定上下文，不因哈希相同跳过渲染"));
    const task = object("tasks", taskId);
    const p = payload(task);
    if (!task) empty("task-summary", "创建或选择任务。创建与补缺不会自动运行。");
    else put("task-summary", pairs([["task_id", task.object_id], ["status", task.status], ["version", task.version], ["content_hash", task.content_hash]]), readable(Object.fromEntries(["context_id", "source_snapshot", "calls", "tools", "tokens", "unknown_calls", "provider_fingerprint", "prediction", "elapsed_s", "error", "pending_action", "awaiting_fields", "lease_id", "lease_until_ms"].map((key) => [key, p[key] ?? null]))));
    const binding = task ? `${task.object_id}:${task.content_hash}` : "";
    if (binding !== supplementBinding) {
      $("supplement-form").reset(); $("reconcile-form").reset(); supplementBinding = binding;
    }
    const requested = Array.isArray(p.awaiting_fields) ? p.awaiting_fields : [];
    const context = payload(object("contexts", p.context_id));
    const fields = ["material_viscosity", "ambient_c"].filter((key) => requested.includes(key) && context[key] === null);
    $("supplement-form").hidden = status(task) !== "awaiting_input";
    $("awaiting-fields").textContent = requested.length ? `服务端要求：${requested.map(label).join("、")}。只补仍为空的允许字段。` : "服务端尚未给出允许补充字段，请重载核查。";
    for (const [key, id] of [["material_viscosity", "material-viscosity"], ["ambient_c", "ambient-c"]]) {
      $(id).disabled = !fields.includes(key) || status(task) !== "awaiting_input";
      $(id).required = fields.includes(key);
    }
    $("supplement-form").dataset.allowedFields = fields.join(",");
    $("reconcile-form").hidden = !canReconcile(task);
    evidenceView("task-evidence", p.evidence);
    candidateView("task-candidate", p.candidate, p.evidence);
  }
  function renderRecommendation() {
    options("recommendation-select", data.recommendations, recommendationId, "请选择推荐，不默认批准", (item) => `${item.object_id} · ${word(item.status)}`);
    const rec = object("recommendations", recommendationId);
    const p = payload(rec);
    const binding = rec ? `${rec.object_id}:${rec.content_hash}` : "";
    if (binding !== reviewBinding) {
      $("review-form").reset(); $("request-execution-form").reset(); reviewBinding = binding;
    }
    if (!rec) empty("recommendation-summary", "尚未选择已提交推荐。提交只是待人工审核，不自动批准或执行。");
    else put("recommendation-summary", pairs([["recommendation_id", rec.object_id], ["status", rec.status], ["version", rec.version], ["content_hash", rec.content_hash], ...["task_id", "context_id", "context_hash", "context_version", "model_hash", "model_version", "constraint_version", "synthetic"].map((key) => [key, p[key]])]), readable(p.approval || p.review || null));
    $("review-binding").textContent = rec ? `审核 / 请求绑定当前完整推荐哈希：${rec.content_hash}。只有 reviewer / admin 可审核。` : "请先选择推荐；审核不触发执行。";
    candidateView("recommendation-candidate", p.candidate, p.evidence);
    evidenceView("recommendation-evidence", p.evidence);
    $("model-draft").textContent = typeof p.untrusted_model_draft === "string" ? p.untrusted_model_draft : "尚无模型解释；不得据此编造候选。";
    const observation = p.observation;
    if (!observation) empty("observation-summary", "尚无成熟观察结果。回读后需满足成熟时间与样本要求，不能把未观察当作改善。");
    else {
      const armRows = ["before", "after"].map((arm) => {
        const sample = observation[arm] || {};
        return [arm === "before" ? "比较前（独立合成采样）" : "比较后（独立合成采样）",
          scalar(sample.defect_count), scalar(sample.sample_count), scalar(sample.defect_rate),
          Array.isArray(sample.rate_ci95) ? sample.rate_ci95.join(" 至 ") : "未提供"];
      });
      const wrap = node("div", undefined, "table-wrap");
      wrap.append(table(["采样组", "缺陷数 / 分子", "样本数 / 分母", "缺陷率", "缺陷率 95% 区间"], armRows,
        "同一冻结上下文的配对模拟比较，两组样本独立采样"));
      put("observation-summary", node("p", "配对合成模拟：before 不是已观测生产基线；不证明现场因果效果。", "safety-inline"),
        wrap, readable(observation));
    }
  }
  function renderExecution() {
    options("execution-select", data.executions, executionId, "请选择执行记录", (item) => `${item.object_id} · ${word(item.status)}`);
    const exec = object("executions", executionId);
    const p = payload(exec);
    const binding = exec ? `${exec.object_id}:${exec.content_hash}` : "";
    if (binding !== executionBinding) { $("apply-form").reset(); executionBinding = binding; }
    if (!exec) { empty("execution-summary", "尚未选择执行记录。"); empty("execution-detail", "尚无设备回读；应用与回读是两个独立操作。"); return; }
    put("execution-summary", pairs([["execution_id", exec.object_id], ["status", exec.status], ["version", exec.version], ["content_hash", exec.content_hash]]));
    const detail = node("div");
    const wrap = node("div", undefined, "table-wrap");
    const rec = object("recommendations", p.recommendation_id);
    const rp = payload(rec);
    const context = boundContext(rp.candidate || {}, rp.evidence);
    wrap.append(table(["参数", "应用前", "冻结目标", "已核验回读值", "单位"], PARAMETERS.map((name) => [
      label(name), scalar(p.before && p.before[name]), scalar(p.target && p.target[name]),
      scalar(p.readback && p.readback.current && p.readback.current[name]), unit(context, name)
    ]), "回读值缺失时绝不以目标值替代"));
    const evidence = node("details");
    evidence.append(node("summary", "来源、请求、设备回执与版本详情"), readable(p));
    detail.append(wrap, evidence);
    if (exec.status === "applied") detail.prepend(node("p", "已应用但尚未回读，不是 executed；请独立核验回执。", "bad"));
    if (exec.status === "unknown") detail.prepend(node("p", "结果未知：禁止再次应用，请用独立回读核验真实回执。", "bad"));
    put("execution-detail", detail);
  }
  function renderAudit() {
    if (!data.audit.length) { empty("audit-list", "当前授权范围内没有审计记录。"); return; }
    put("audit-list", table(["时间", "对象类别 / ID", "操作", "可读详情"], data.audit.map((event) => {
      const detail = node("details"); detail.append(node("summary", "查看审计字段"), readable(event.detail));
      return [stamp(event.created_at_ms), `${event.kind} / ${event.object_id}`, word(event.action), detail];
    }), "仅本人及当前授权工厂范围，最多 100 条；不包含令牌"));
  }
  function canRun(task) {
    // Only an explicit click asks the backend to recover a stranded running task.
    // Server lease/unknown-call guards decide; browser time is not a safety authority.
    if (status(task) === "running") return true;
    const p = payload(task);
    if ((typeof p.unknown_calls === "number" && p.unknown_calls > 0) || /unknown/.test(p.error || "")) return false;
    if (status(task) === "created") return true;
    return status(task) === "blocked" && p.unknown_calls === 0 &&
      !/budget_exhausted|source_changed|provider_changed/.test(p.error || "");
  }
  function canReconcile(task) {
    return !!me && me.role === "admin" && ["blocked", "running", "cancelled"].includes(status(task)) &&
      typeof payload(task).unknown_calls === "number" && payload(task).unknown_calls > 0;
  }
  function controls() {
    updateBusy();
    const task = object("tasks", taskId);
    const rec = object("recommendations", recommendationId);
    const exec = object("executions", executionId);
    const locked = !me || !stateReady || busy.has("reload");
    const inFlight = (key) => locked || busy.has(key);
    const mutationBusy = [...busy].some((key) => key !== "reload");
    const actionLocked = locked || mutationBusy;
    $("create-task").disabled = actionLocked || !object("contexts", contextId);
    $("run-task").textContent = status(task) === "running" ? "显式请求运行态恢复（后端核验租约）" : "显式运行 / 恢复任务";
    $("run-task").disabled = actionLocked || !canRun(task);
    // A cancellation must remain independently sendable during a long run request.
    $("cancel-task").disabled = inFlight("cancel") || [...busy].some((key) => !["run", "cancel"].includes(key)) || !["created", "running", "awaiting_input", "blocked"].includes(status(task));
    $("supplement-submit").disabled = actionLocked || status(task) !== "awaiting_input" || !$("supplement-form").dataset.allowedFields;
    $("reconcile-submit").disabled = actionLocked || !canReconcile(task) || !$("reconcile-confirm").checked;
    const mayReview = !actionLocked && me && ["admin", "reviewer"].includes(me.role) && status(rec) === "pending_review";
    $("review-decision").disabled = !mayReview;
    $("review-note").disabled = !mayReview;
    $("review-submit").disabled = !mayReview || !["approved", "rejected"].includes($("review-decision").value) || !$("review-note").value.trim();
    $("execution-confirm").disabled = actionLocked || status(rec) !== "approved";
    $("request-execution").disabled = $("execution-confirm").disabled || !$("execution-confirm").checked;
    $("apply-confirm").disabled = actionLocked || status(exec) !== "pending";
    $("apply-execution").disabled = $("apply-confirm").disabled || !$("apply-confirm").checked;
    $("readback-execution").disabled = actionLocked || !["applied", "unknown"].includes(status(exec));
    $("observe-recommendation").disabled = actionLocked || status(rec) !== "executed";
    $("reload-button").disabled = !me || busy.has("reload");
    ["context-select", "task-select", "recommendation-select", "execution-select"].forEach((id) => { $(id).disabled = locked; });
  }
  function render() { renderContext(); renderTask(); renderRecommendation(); renderExecution(); renderAudit(); controls(); }
  function selectRelatedTask(id) {
    taskId = id;
    const task = object("tasks", id);
    contextId = task ? payload(task).context_id : "";
    const rec = id ? data.recommendations.find((item) => payload(item).task_id === id) : null;
    recommendationId = rec ? rec.object_id : "";
    executionId = rec ? relatedExecution(rec.object_id) : "";
  }
  function relatedExecution(id) {
    const exec = data.executions.find((item) => payload(item).recommendation_id === id || item.object_id === `exec-${id}`);
    return exec ? exec.object_id : "";
  }
  async function detail(kind, id) {
    if (!id || !me) return;
    const expectedEpoch = epoch;
    const selectedAtStart = { tasks: taskId, recommendations: recommendationId, executions: executionId }[kind];
    const read = {};
    pendingReads.add(read); updateBusy();
    try {
      const entry = await api(`/api/${kind}/${encodeURIComponent(id)}`, undefined, expectedEpoch);
      if (expectedEpoch !== epoch) return;
      const selectedNow = { tasks: taskId, recommendations: recommendationId, executions: executionId }[kind];
      if (selectedNow !== selectedAtStart) return;
      accept(entry); render();
    } catch (error) { if (expectedEpoch === epoch) showError(error); }
    finally { pendingReads.delete(read); updateBusy(); }
  }
  function selectTask(id) {
    // Always rebind and render, even if a previously visited object's hash is unchanged.
    resetForms(); selectRelatedTask(id); clearError(); render();
    void detail("tasks", id);
  }
  function selectRecommendation(id) {
    resetForms(); recommendationId = id;
    const rec = object("recommendations", id);
    taskId = payload(rec).task_id || "";
    contextId = payload(rec).context_id || "";
    executionId = id ? relatedExecution(id) : "";
    clearError(); render(); void detail("recommendations", id);
  }
  $("login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const entered = $("token").value.trim();
    if (!entered || $("login-button").disabled) return;
    invalidate(); token = entered; $("token").value = ""; clearError();
    const expectedEpoch = epoch;
    busy.add("login"); updateBusy();
    $("login-button").disabled = true;
    showMessage("正在验证本地身份与合成服务。");
    try {
      const [identity, health] = await Promise.all([api("/api/me", undefined, expectedEpoch), api("/health", undefined, expectedEpoch)]);
      if (expectedEpoch !== epoch) return;
      if (health.synthetic_only !== true) throw new Error("服务未明确声明 synthetic_only，已拒绝连接。");
      me = identity;
      await refresh(expectedEpoch);
      if (expectedEpoch !== epoch) return;
      $("login-form").hidden = true; $("session-controls").hidden = false; $("workbench").hidden = false;
      $("session-status").textContent = `${identity.user_id} · ${word(identity.role)} · 工厂 ${(identity.plants || []).join("、")} · ACL ${identity.acl_version}`;
      const fingerprint = typeof health.provider === "string" ? health.provider : "本地提供方（详情由服务端管理）";
      $("provider-status").textContent = `仅合成 · ${fingerprint}`;
      showMessage("已连接。请选择上下文或恢复既有任务，不自动运行任何动作。");
    } catch (error) {
      if (expectedEpoch === epoch) { logout("连接未完成，身份与页面数据已清空。"); showError(error); }
    } finally {
      if (expectedEpoch === epoch) {
        busy.delete("login"); $("login-button").disabled = false; controls();
      }
    }
  });
  $("logout-button").addEventListener("click", () => logout());
  window.addEventListener("pagehide", () => logout("页面已离开，内存身份已清空，请重新登录。"));
  $("reload-button").addEventListener("click", async () => {
    if (!me) return;
    invalidate(); resetForms(); stateReady = false;
    const expectedEpoch = epoch;
    busy.add("reload"); clearError(); render();
    showMessage("已中止旧浏览器请求，正在重载。服务端运行不因此取消。");
    try {
      const identity = await api("/api/me", undefined, expectedEpoch);
      if (expectedEpoch !== epoch) return;
      me = identity;
      await refresh(expectedEpoch);
      if (expectedEpoch === epoch) showMessage("已重载服务端状态；旧请求不会恢复旧页面。请重新核对并确认人工动作。");
    } catch (error) { if (expectedEpoch === epoch) showError(error); }
    finally { if (expectedEpoch === epoch) { busy.delete("reload"); render(); } }
  });
  $("context-select").addEventListener("change", () => {
    contextId = $("context-select").value; taskId = recommendationId = executionId = "";
    resetForms(); clearError(); render();
  });
  $("task-select").addEventListener("change", () => selectTask($("task-select").value));
  $("recommendation-select").addEventListener("change", () => selectRecommendation($("recommendation-select").value));
  $("execution-select").addEventListener("change", () => {
    const id = $("execution-select").value;
    const exec = object("executions", id);
    const recId = payload(exec).recommendation_id || (id.startsWith("exec-") ? id.slice(5) : "");
    selectRecommendation(recId); executionId = id; render(); void detail("executions", id);
  });
  $("create-task").addEventListener("click", () => {
    if ($("create-task").disabled) return;
    const id = contextId;
    void perform("create", async (e) => {
      const task = await api("/api/tasks", { action_id: actionId(), context_id: id }, e);
      accept(task); if (contextId === id) selectRelatedTask(task.object_id);
    }, "任务已创建，尚未运行。请核对后显式运行。");
  });
  $("run-task").addEventListener("click", () => {
    if ($("run-task").disabled) return;
    const id = taskId;
    void perform("run", async (e) => {
      const task = await api(`/api/tasks/${encodeURIComponent(id)}/run`, {}, e);
      accept(task);
      if (taskId === id && task.status === "submitted") {
        const rec = await api(`/api/recommendations/${encodeURIComponent(`rec-${id}`)}`, undefined, e);
        accept(rec);
        if (taskId === id) recommendationId = rec.object_id;
      }
    }, "运行请求已返回真实任务状态。提交的推荐仍需独立人工审核。");
  });
  $("cancel-task").addEventListener("click", () => {
    if ($("cancel-task").disabled) return;
    const id = taskId;
    void perform("cancel", async (e) => accept(await api(`/api/tasks/${encodeURIComponent(id)}/cancel`, { action_id: actionId() }, e)), "取消请求已返回，请以任务状态与用量记录为准。");
  });
  $("supplement-form").addEventListener("submit", (event) => {
    event.preventDefault(); if ($("supplement-submit").disabled) return;
    const fields = {};
    for (const [key, id] of [["material_viscosity", "material-viscosity"], ["ambient_c", "ambient-c"]]) {
      if (!$(id).disabled && $(id).value !== "") fields[key] = Number($(id).value);
    }
    const note = $("supplement-note").value.trim();
    if (!Object.keys(fields).length || !note || Object.values(fields).some((value) => !Number.isFinite(value)) || !$("supplement-form").reportValidity()) return;
    const id = taskId;
    void perform("supplement", async (e) => accept(await api(`/api/tasks/${encodeURIComponent(id)}/supplement`, { action_id: actionId(), fields, note }, e)), "补缺已返回；旧派生证据不再有效，未自动运行，请核查新上下文版本。");
  });
  $("review-form").addEventListener("submit", (event) => {
    event.preventDefault(); if ($("review-submit").disabled || !$("review-form").reportValidity()) return;
    const rec = object("recommendations", recommendationId);
    const decision = $("review-decision").value;
    const note = $("review-note").value.trim();
    if (!rec || !note || !["approved", "rejected"].includes(decision)) return;
    void perform("review", async (e) => accept(await api(`/api/recommendations/${encodeURIComponent(rec.object_id)}/review`, { action_id: actionId(), recommendation_hash: rec.content_hash, decision, note }, e)), "人工审核已返回。批准不会自动创建执行请求或应用参数。");
  });
  $("request-execution-form").addEventListener("submit", (event) => {
    event.preventDefault(); if ($("request-execution").disabled || !$("execution-confirm").checked) return;
    const rec = object("recommendations", recommendationId);
    void perform("request", async (e) => {
      const exec = await api(`/api/recommendations/${encodeURIComponent(rec.object_id)}/request-execution`, { action_id: actionId(), recommendation_hash: rec.content_hash, confirmed: true }, e);
      accept(exec); if (recommendationId === rec.object_id) executionId = exec.object_id;
    }, "合成执行请求已创建，尚未应用设备参数。请单独确认应用。");
  });
  $("apply-form").addEventListener("submit", (event) => {
    event.preventDefault(); if ($("apply-execution").disabled || !$("apply-confirm").checked) return;
    const id = executionId;
    void perform("apply", async (e) => accept(await api(`/api/executions/${encodeURIComponent(id)}/apply`, { action_id: actionId() }, e)), "合成应用请求已返回；applied 不等于回读成功。unknown 禁止重发，请独立回读。");
  });
  $("readback-execution").addEventListener("click", () => {
    if ($("readback-execution").disabled) return;
    const id = executionId;
    void perform("readback", async (e) => accept(await api(`/api/executions/${encodeURIComponent(id)}/readback`, { action_id: actionId() }, e)), "回读请求已返回；仅匹配真实回执才确认 executed，观察仍是下一独立阶段。");
  });
  $("observe-recommendation").addEventListener("click", () => {
    if ($("observe-recommendation").disabled) return;
    const id = recommendationId;
    void perform("observe", async (e) => accept(await api(`/api/recommendations/${encodeURIComponent(id)}/observe`, { action_id: actionId() }, e)), "观察请求已返回；请查看分子、分母、差值、不确定性与真实合成结论，不证明现场因果。");
  });
  $("reconcile-form").addEventListener("submit", (event) => {
    event.preventDefault(); if ($("reconcile-submit").disabled || !$("reconcile-form").reportValidity()) return;
    const id = taskId;
    const note = $("reconcile-note").value.trim();
    const raw = $("reconcile-tokens").value;
    const tokens = raw === "" ? null : Number(raw);
    if (!note || (tokens !== null && (!Number.isInteger(tokens) || tokens < 0 || tokens > 100000))) return;
    const call_id = $("reconcile-call").value.trim();
    void perform("reconcile", async (e) => accept(await api(`/api/tasks/${encodeURIComponent(id)}/reconcile`, { action_id: actionId(), call_id, no_result: true, tokens, note }, e)), "模型无结果用量对账已返回；不代表设备已执行，不自动重试模型。");
  });
  ["review-decision", "review-note", "execution-confirm", "apply-confirm", "reconcile-confirm"].forEach((id) => {
    $(id).addEventListener("input", controls); $(id).addEventListener("change", controls);
  });
  render();
})();
