# 架构与数据语义

## 1. 项目边界

这是**本地合成工艺参数推荐 Demo／测试版**，用于生产部署前的本地验证，不是生产模型或设备规范。上下文、经验卡、专家示范、设备和质量过程均为合成内容。系统不提供生产设备适配器，不保证现场效果，也不作真实设备 exactly-once 承诺。

必须分开理解三个组件：

- **工具选择提供方**：为单 Agent 选择下一工具。真实模式使用明确配置的本地模型；`--test-mode` 使用确定性的 `SyntheticProvider`，它不是 LLM，不能作为真实模型自主决策的验收结果。
- **行为克隆 BC**：真正通过 NumPy 训练的多头神经网络，模仿合成专家的参数动作。即使选择 TEST 工具提供方，BC 仍需独立训练、注册和实际推理。
- **合成质量模拟器**：根据上下文与参数计算期望缺陷概率，并作二项采样。它既不是 BC 的输出，也不是生产现场观察。

本文件描述当前代码；操作步骤见 [usage.md](usage.md)。接口约束见 [implementation-contract.md](implementation-contract.md)、[storage-contract.md](storage-contract.md)、[service-contract.md](service-contract.md) 和 [api-contract.md](api-contract.md)。不在本文预写模型指标、验证通过率或固定改善结论。

## 2. 分层与实际调用关系

| 文件 | 责任与边界 |
| --- | --- |
| `domain.py` | 严格输入模型、规范 JSON、哈希、Decimal 四参数算术、上下文和候选硬约束 |
| `fixtures.py` | 原创上下文、专家示范生成器、可计算合成质量过程 |
| `bc.py` | 真正训练、分组划分、校准、留出评测、JSON 模型核验与预测拒绝 |
| `storage.py` | SQLite 事务、身份与 ACL、owner/plant 隔离、对象 CAS、内容寻址制品、幂等回执和审计 |
| `service.py` | 上下文与设备来源绑定、本人当前模型、经验检索、补缺、独立重算后提交推荐 |
| `agent.py` | 单 Agent 三节点 LangGraph、七工具白名单、预算、租约、恢复、取消和调用用量账本 |
| `provider.py` / `managed_provider.py` | 显式 TEST / 本地工具决策提供方及本地托管进程边界；失败不自动回退云端或 TEST |
| `actions.py` | 图外人工审核、合成执行请求、设备应用、独立回读、成熟观察 |
| `api.py` / `schemas.py` | 数字回环 HTTP 边界、Bearer 身份、严格请求模型；不接受模型给出的任意工具或设备地址 |
| `cli.py` | 显式初始化、真实训练、启动本地服务；不自动创建推荐或执行设备 |
| `templates/index.html` / `static/app.js` | 当前中文工作台；按对象状态逐步启用操作，不在浏览器持久化令牌 |

调用概览：

```text
本地 CLI：init / train / serve
                   |
浏览器 -> 回环 API -> Store 身份与作用域检查
                   |
                   +-> AgentEngine -> RecommendationService / BC / 硬约束
                   |                     |
                   |                     +-> pending_review 推荐
                   |
                   +-> ActionService：审核 -> 请求 -> 合成应用 -> 回读 -> 观察
                                         |
                              SQLite 对象 + JSON 制品 + 审计
```

图外动作不是 Agent 工具。七工具中没有批准、应用设备或宣布效果成功的权限。

## 3. 实际三节点图与七工具

`agent.py` 实际注册的业务节点只有 **`agent`、`guard`、`tools`**。`START`/`END` 是图的边界，不是另外两个业务节点；页面的五个工作区也不是图节点。

```text
START -> agent --有下一动作--> guard -> tools --仍在运行--> agent
           |                            |
           +--无可用安全动作--> END      +--提交/补问等停止--> END
```

- `agent`：读取持久化任务，检查租约、来源和预算；优先恢复已落盘且未消费的回复，否则先写 `calls.started` 再调用提供方。模型调用期间不持有 SQLite 写事务。每次只接受一个结构化工具动作。
- `guard`：重新核验身份、任务状态、租约、预算、当前来源、回复与调用记录的绑定，以及工具和参数是否属于当前菜单。
- `tools`：再次执行相同边界核验，完成一个受控工具，写入结果、证据、消费标记与审计。涉及状态的修改使用同事务与对象版本 CAS。

模型只在当时可用的菜单中选择，不获得任意 Python、SQL、URL 或文件工具。依赖关系会限制菜单：先取上下文；有缺失先补问；经验与 BC 可在可用时选择；二者完成后才能草拟、检查和提交。菜单约束不是模型生成参数的许可。

| 工具 | 实际作用 | 是否产生任务 Evidence |
| --- | --- | --- |
| `get_process_context` | 读取绑定上下文并列出缺失、时间、测量等问题 | 是 |
| `retrieve_experience` | 按 defect/material/process 匹配代码内原创经验卡；不相关时为空 | 是 |
| `predict_bc` | 使用本人该工厂已注册模型实际推理；没有模型或不安全则拒绝动作 | 是 |
| `draft_recommendation` | 从实际 BC 结果复制 `raw_delta`，不接受模型自编参数 | 否，保存 `draft` |
| `check_constraints` | 独立检查原始与量化后候选，冻结来源和候选哈希 | 是 |
| `request_human_input` | 只请求当前缺失的黏度/环境温度，进入 `awaiting_input` | 否，保存问题与待补字段 |
| `submit_recommendation` | 引用完整 Evidence 与精确候选哈希；服务重算通过后创建唯一 `pending_review` 推荐 | 否，保存推荐关联 |

任务状态为 `created`、`running`、`awaiting_input`、`blocked`、`submitted`、`no_recommendation`、`cancelled`。`submitted` 只表示推荐已提交，不表示任何设备动作。

## 4. BC：条件多头动作模仿，不是效果预测

当前特征为四个原参数、材料黏度、环境温度、严重程度，以及四类缺陷的 one-hot，共 11 维。MLP 共享主干为 `11 -> 48 -> 32`，输出两个头：

1. **组合分类头**：11 个 mask 类，覆盖不动作、单参数和双参数组合。
2. **带符号增量回归头**：四个增量，经过 `tanh` 与各轴尺度转换；分类 mask 将未选参数的增量置零。方向由增量导出，不存在一个另行训练的“效果头”或“方向头”。

这里的“条件”是以工艺上下文为条件预测动作。分类结果筛选回归输出；它不意味着系统学到了干预后的因果结果。

示范专家枚举有限合法动作，用已知原创模拟期望和动作成本选择标签。默认生成 120 个设备/批次血缘组，每组 12 条记录。训练按完整 `group_id` 作 60%/20%/20% 的 train/calibration/test 划分，同一设备血缘不得跨组。归一化只拟合训练集，温度校准只用校准集，测试在权重与校准冻结后评估，不用于调参。

预测检查上下文、训练支持范围、材料等类别来源和置信度。当前固定置信度阈值为 0.55；候选仍需独立通过硬约束。置信度是经校准的**组合动作分类置信度**，不是缺陷概率，更不是“改善概率”。正常工况的零动作不能成为可执行候选，拒绝或无推荐是正常结果。

`train` 分别保存模型和绑定该 `model_hash` 的评测：训练损失、划分血缘、校准、组合准确率、增量误差、覆盖率及无动作/简单规则等合成对照。`model_metrics` 是独立对象，不把评测伪装成模型权重。`GET /api/models` 返回模型摘要与绑定评测，不返回权重；缺少绑定评测时明确标记测试数据不可用。

## 5. Decimal 硬约束与来源冻结

四参数与所有候选增量的公共 JSON 值都是**普通十进制字符串**，例如 `"204"`、`"4.250000"`，不是 float、布尔、百分数或指数形式。领域运算使用 Decimal，不靠二进制浮点或 clamp 把非法原始建议修成合法建议。

| 参数键 | 单位 | 允许范围 | 网格 | 单次最大绝对增量 |
| --- | --- | --- | --- | --- |
| `temperature` | `degC` | 180–240 | 1 | 6 |
| `speed` | `mm/s` | 20–100 | 5 | 10 |
| `pressure` | `MPa` | 30–70 | 1 | 4 |
| `cooling` | `s` | 8–30 | 1 | 3 |

原始目标与增量先过边界、方向、最多两轴和耦合检查，再按最近目标网格、半点取偶数（`ROUND_HALF_EVEN`）量化，并重新检查。短射只允许温度/速度/压力非负变化；飞边只允许速度/压力非正变化；翘曲只允许温度非正、冷却非负变化；正常只允许不变。温度至少 230 时冷却至少 15，速度大于 80 时压力不大于 60。可执行候选必须至少一项非零。

上下文还要求正确单位、受支持工艺版本、测量合格、至少 30 个原样本、非未来且不超过 24 小时。缺失黏度或环境温度不得推测。人工补缺只填原先为 `null` 的两项并记录依据，提升上下文 revision，更新设备来源绑定；不改变设备参数、不刷新观测时间，不重置任务预算，清空旧派生证据与候选。

推荐绑定上下文 hash/对象版本、模型 hash/对象版本/对象内容 hash、经验和约束版本。即使重新注册同一模型内容，模型对象版本增加也会使旧任务来源失效。执行请求和实际应用均再次检查来源、时效、候选与设备 active context、原值和版本。

### 哈希不是密码学签名

规范 JSON 使用有界结构、排序键、紧凑编码与有限数值，再计算 SHA256。Evidence 的 `content_hash` 是 Evidence 除自身哈希字段的内容摘要；候选与模型哈希同样排除各自自引用字段。

哈希能标识内容、检测变化，**没有秘密签名密钥，不证明作者身份，也不是电子签章**。即使代码方法或界面称“签发”，信任来源仍是：后端在已授权任务中生成并保存证据，提交时匹配该任务中的真实证据，并重算上下文、检索、BC 和硬约束。模型不能通过自算哈希签发可信 Evidence。

## 6. 身份、事务、恢复与两类 unknown

### 身份与存储

- `Principal` 来自服务端鉴权，包含 user/role/plants/acl_version；每次操作重新检查用户仍激活、角色、工厂与 ACL 版本。
- 对象和审计按 **owner + 当前授权 plant** 隔离；`admin`/`reviewer` 也不会自动看到其他 owner 的对象。审核/模型注册要求 reviewer 或 admin；已获授权的 owner 可以进行后续合成执行。只有 admin 可做调用用量对账。
- SQLite 写操作用 `BEGIN IMMEDIATE`，对象更新以 `expected_version` 做 CAS。关联修改同事务；推理调用在事务外，返回后重新检查取消、授权、租约与来源。
- `action_id` 的幂等作用域是 `(user_id, action_id)`，请求摘要绑定阶段、目标和完整业务内容。同 key 不同 body/目标/阶段拒绝。回放前仍检查身份与目标权限；历史回执不是把设备恢复到旧值的命令。
- 数据存于显式本地目录的 `data.sqlite3` 与 `artifacts/<sha256>.json`。制品有大小、结构、完整性与文件路径检查，不使用 pickle。令牌只存 hash；数据库、模型、日志、凭据与运行目录不属于应上传的源码内容。

### 模型调用结果/用量 unknown

任务预算当前为最多 14 次模型调用、12 次工具、32000 tokens、420 秒累计执行耗时；租约 900 秒。租约是恢复和并发保护，不是免费追加预算或生产作业锁。

`calls` 保存每次调用的 `started/completed/failed/unknown`、tokens、耗时、回复制品与错误码。已明确未发送的失败可以计 0；可靠已知的用量即使回复无效、任务取消或期间撤权也要记账。已发送但用量未知用 `tokens=NULL`，不是零。任务摘要 `tokens` 只是已知/已对账的数值合计，必须结合 `unknown_calls` 阅读；后者包括尚在 `started` 和处于 `unknown` 的记录。

**已知 tokens 不等于调用结果已确定。** 已发送调用的结果或清理状态仍不明确时，即使拿到可靠 tokens，也保留 `unknown`，不能仅凭用量非空解除阻塞、自动重试或执行工具。

任务以 `accounted_call_elapsed_s` 保存已计入任务耗时的调用账本累计耗时水位。读取时只把账本累计耗时超过该水位的新增部分加入 `elapsed_s`，再推进水位；任务保存时一起持久化。运行中的累计计时包含进入本轮前的累计值、本轮实际耗时，以及本轮开始后旧租约调用补记的新增耗时，再与已保存值取较大值，不能把旧调用迟到消耗吞入本轮耗时。发送前重新读取账目、授权与来源；异常停止时，耗时采样和任务保存处于同一写事务。这样恢复读取不会反复叠加同一笔已计入的调用耗时，取消后或迟到回复补记的耗时也不会被当作免费消耗；该水位不是观测成熟时间或可重置预算。

未知调用禁止自动重试。本人任务的管理员只可核实“没有可用结果”并对账：已知 tokens 不得减少；仍未知则保守占用剩余 token 预算。租约仍有效时拒绝对账；租约已失效或不存在时，遗留 `started` 调用可先转为 `unknown/interrupted_call` 再对账，包括已经 `cancelled` 的任务。对账后调用记为 `failed/reconciled_no_result`，清除待处理动作；已取消任务仍为 `cancelled`，其他可对账任务回到 `created` 也不会自动运行，后续仍受预算、来源和授权约束。该操作不能补造模型回复，不解除托管进程的 cleanup-unknown 锁。

对于已按“无可用结果”对账的调用，原工作者迟到回复只能分别按较大值补记已保存的 tokens 和耗时，不能减少成本、把调用改回 `completed` 或恢复可执行动作。任务取消、终态与租约核验仍阻止旧工作者继续提交工具。UI 对 admin 且 `unknown_calls > 0` 的 `running`、`blocked`、`cancelled` 任务显示对账表单；遗留 `running` 任务允许显式请求后端恢复检查。页面不凭浏览器时钟判定租约、不自动重试，后端仍拒绝有效租约和未对账的未知结果。

取消会阻断后续工具提交，但不撤销已发生的模型资源消耗，也不自动结清遗留调用；浏览器退出、刷新或停止等待不是取消服务端任务。已提交终态任务不能用任务取消撤销图外推荐。

### 合成设备 ACK unknown

这是不同于模型账本的另一种 unknown：设备可能已更新，但调用方未得到 ACK。`apply_synthetic` 分两次事务：

1. 重验请求与设备，实际更新本地设备参数和 revision，创建 `device_receipt`，持久化 execution/recommendation 的保守 `unknown`，同时保留全局 action_id 幂等记录。
2. 第一事务提交后，正常路径在独立事务确认本地 ACK，将 execution 置 `applied`。内部测试故障 `after_apply` 则停在已提交的 `unknown`；HTTP/UI 不开放该故障参数。

内部 `action_result` 对象保存可最终化的应用响应，避免修改不可变的通用幂等回执。进程中断、撤权或设备漂移不能回滚已经提交的设备事实；unknown 不盲重发。恢复只有独立回读一条路：读取实际保存的设备回执和当前设备，核对 request、target、before、来源、身份、hash、版本和时间。缺失或不匹配就保持 unknown，不能输入“我确认成功”替代事实。

## 7. 图外状态与观察含义

**“独立人工审核”是与 Agent 提交、执行请求分离的一次显式、已授权的人工动作，不是要求不同人员审批的双人复核（four-eyes）制度。** 当前 owner + plant 隔离下，推荐 owner 在拥有该工厂 reviewer/admin 权限时可审核自己的推荐；其他用户即使具有 reviewer/admin 角色，也不能审核该 owner 的对象。operator 不因此获得审核权限，系统也没有跨 owner 转派审批流程。

| 操作完成后 | recommendation 状态 | execution 状态 | 已证明/尚未证明 |
| --- | --- | --- | --- |
| Agent 提交 | `pending_review` | 不存在 | 冻结安全候选；尚未审核 |
| 人工批准 / 拒绝 | `approved` / `rejected` | 不存在 | 记录决定；不启动设备 |
| 显式请求执行 | `execution_pending` | `pending` | 请求已保存；尚未更新设备 |
| 本地应用并获得 ACK | `execution_pending` | `applied` | 合成设备已更新；尚未独立回读 |
| 应用后 ACK 不明确 | `unknown` | `unknown` | 不猜测成功，不重新应用 |
| 回执与实际设备匹配 | `executed` | `executed` | 独立回读已确认 |
| 成熟采样完成 | `observed` | 仍为 `executed` | 有合成比较结果；不证明现场因果 |

审核需当前推荐的精确 Store `content_hash`。审核记录追加到 payload 后 hash 改变，执行请求必须使用**批准后新的当前 hash**，不是审核前 hash，更不是 candidate_hash。`confirmed` 必须为真正的布尔 `true`，数字 `1`/`1.0` 不算确认。状态与 CAS 独立检查；对象元数据不是 payload 哈希的一部分。

观察要求从有效 `readback_at_ms` 起真实服务端时钟至少经过 60000ms。之前等待、页面计时或修改输入不能替代这个条件。观察前再次检查回执、设备当前目标/版本和原上下文来源。服务层支持严格整数样本数 60–10000；当前 HTTP/UI 固定使用每组 120 和默认种子 27183，后组使用不同种子，不开放 seed 或样本刷选。

`before` 与 `after` 是同一冻结上下文、应用前/目标参数下**新计算的两组独立合成样本**。“配对”指上下文配对，不是同一批真实产品前后重复测量。before 不是既有生产基线。保存缺陷数、样本数、缺陷率、Wilson 边际区间与 Newcombe 独立二项差值区间。差值采用“后 − 前”：整个区间小于 0 才标记 `synthetic_observed_improvement`，整个区间大于 0 标记 `worsened`，其余为 `no_clear_improvement`。固定保留 `causal_effect_proven=False`。

## 8. 关键数据字典

### 通用对象与版本

| 键 | 含义 |
| --- | --- |
| `kind`, `object_id` | 对象类别和不可混用的标识；推荐为 `rec-<task_id>`，执行为 `exec-<recommendation_id>` |
| `owner`, `plant_id` | 服务端绑定的所属用户与工厂，不由浏览器正文覆盖 |
| `status` | 该类别的业务状态；不能把 task 的 `submitted` 当作 execution 的 `executed` |
| `version` | Store 对象整数 CAS 版本，每次对象更新递增，即使 payload hash 没变 |
| `content_hash` | 保存的完整 payload 的 SHA256，不包含外层 status/version；API 模型摘要是投影，不能用投影重算权重对象 hash |
| `payload` | 该类别的领域数据，不含可执行代码 |
| `revision` | 上下文自身修订或设备实际参数修订，依所在对象区分，不等于 Store `version` |
| `process_version`, `constraint_version`, `experience_version` | 各来源的语义版本标识，不是 CAS 计数 |
| `*_at_ms`, `lease_until_ms` | UTC epoch 毫秒；`elapsed_s` 是累计耗时秒 |

### 来源、证据和候选

| 键 | 含义 |
| --- | --- |
| `context_id`, `context_hash`, `context_version` | 上下文标识、完整上下文摘要、上下文 Store 对象版本 |
| `model_hash` | 模型 JSON 除自身该字段的摘要 |
| `model_content_hash`, `model_version` | 来源快照中的模型 Store 对象 payload hash 与整数版本；模型制品内部 `model.model_version` 另指格式版本字符串 |
| `source_snapshot` | 任务创建时冻结的上下文/模型/经验/约束来源；不等同于当前最新来源 |
| `evidence_id`, `tool`, `payload`, `content_hash` | 后端生成的证据标识、产生工具、实际结果与内容摘要；绑定 task/context |
| `raw_delta` | BC 原始四参数增量，十进制字符串，不允许人工文本替换 |
| `delta`, `target`, `directions` | 量化增量、目标值和由增量推导的方向 |
| `eligible`, `issues` | 当前硬约束资格与具体问题；置信度高不覆盖问题 |
| `candidate_hash` | 完整冻结候选排除自身字段后的摘要，绑定 task/context/model |
| `untrusted_model_draft` | 未审核模型解释，普通文本，不是权威参数、审批或设备事实 |

### 执行、观察与账本

| 键 | 含义 |
| --- | --- |
| `action_id` | 用户一次业务操作的幂等键；不是设备 request_id 或模型 call_id |
| `request_id`, `request_hash` | 执行请求稳定标识与冻结内容摘要；执行对象内 request_id 等于 execution_id |
| `recommendation_hash` | 操作所绑定的推荐 payload hash；审核和请求对应不同时间点的精确值 |
| `before`, `target`（execution） | 原设备值与冻结目标，不代表已经成功回读 |
| `expected_device_version`, `expected_device_revision` | 请求时绑定的设备 Store 版本与参数修订，应用前必须仍相同 |
| `device_receipt_hash` | 实际本地合成设备事务产生的回执制品 hash，不是人工声明 |
| `readback.current`, `readback_hash`, `readback_at_ms` | 经真实回执及设备核验后的参数、回读摘要与成熟计时起点 |
| `before`, `after`（observation） | 两组新生成的模拟采样结果，区别于 execution 中的参数字典 |
| `defect_count`, `sample_count`, `defect_rate` | 缺陷分子、样本分母、两者之比；上下文 sample_count 与观察新样本数不是同一计数 |
| `expected_defect_rate` | 模拟器计算的期望概率，不是 BC 置信度，也不是观察频率 |
| `difference_after_minus_before`, `difference_ci95` | 后减前的缺陷率差和二项采样不确定性区间 |
| `baseline_is_observed_production`, `causal_effect_proven` | 均为 false，明确否认生产基线与现场因果证明 |
| `call_id`, `lease_id` | 模型调用标识与工作者租约绑定，不是设备请求 |
| `tokens`, `unknown_calls` | 已知/已对账 token 数与仍为 `started/unknown` 的调用数；NULL 不是零，tokens 已知也不代表结果已确定 |
| `calls.elapsed_s` | 单次模型调用已记账耗时；无结果对账后的迟到回复只能增加该值 |
| `elapsed_s`（task） | 任务累计预算耗时，吸收调用账本新增耗时，并与运行计时取较大值；不是观察成熟年龄 |
| `accounted_call_elapsed_s`（task） | 已并入任务耗时的调用账本累计水位，防止恢复读取重复加算同一笔耗时 |
| `error_code=reconciled_no_result`（calls） | 已核实无可用结果并完成用量对账；保留 `failed`，迟到回复不恢复动作 |

## 9. HTTP 与当前界面的限制

服务默认绑定 `127.0.0.1:8005`，拒绝非数字回环 Host、跨源写入、查询参数凭据、重复 JSON 键和超限/非有限输入。写请求使用严格 JSON，未知字段拒绝。页面令牌只在内存中，以 Authorization 头发送；不进入 URL、Cookie、localStorage 或 sessionStorage。模型解释通过文本节点显示，不作为 HTML 执行。

当前界面提供上下文选择、模型评测摘要、任务创建/运行/取消/恢复、条件显示的补缺和管理员对账、推荐审核、独立执行请求/应用/回读/观察及本人审计。没有网页训练、模型上传、任意参数编辑、任意上下文导入、生产设备地址、故障注入、seed 选择或完整 calls 明细浏览器接口。页面分步确认只是交互保护，后端仍逐步重新鉴权和核验。
