# 参数推荐实施契约（本地 Demo／测试版）

本文件定义本地测试版本的接口，用于生产部署前的本地验证，不代表生产模型或设备规范。输入、知识、专家策略、设备与效果均为原创合成内容。默认本地模型，禁止生产写入。

## 分层与文件职责

- `domain.py`：严格上下文/参数/候选/硬约束、精确 Decimal 算术；`fixtures.py`：原创示范与可计算合成过程。
- `bc.py`：实际 NumPy MLP 多头分类/回归训练、安全 JSON 模型制品、校准、分组留出评测、OOD 与来源检查。
- `storage.py`：身份/授权、SQLite 事务、制品及审计；`service.py`：上下文/模型/经验/推荐来源整合。
- `agent.py`：LangGraph 单 Agent 七工具 ReAct + 有限预算、租约/CAS、恢复、补问、取消和未知用量对账。
- `actions.py`：图外审核、合成设备执行/回读/未知对账与成熟效果观察。
- `provider.py` / `managed_provider.py`：明确 TEST 或真实回环模型，不回退云端。
- `api.py` / `cli.py` / `static/` / `templates/`：本地用户入口，不保留浏览器 token。

## 公共值与领域 API

`errors.py`: `DomainError(code: str, message: str, status: int = 409)`，属性同名。

`domain.py` 必须提供：

- `canonical(value) -> bytes`：有界严格 JSON 所需 UTF-8、sorted keys、ensure_ascii=False、allow_nan=False、紧凑 separators；`digest(value) -> str`：SHA256。
- `Identifier`：1–80 字符 `[A-Za-z0-9][A-Za-z0-9_.-]{0,79}`。
- `StrictModel`：Pydantic `extra='forbid', strict=True, allow_inf_nan=False`。
- `ParameterValues`：四个严格十进制字符串字段 `temperature`, `speed`, `pressure`, `cooling`。禁止布尔、float 和指数形式，有限长度。所有公共 JSON 使用字符串，不输出 Decimal。
- `ProcessContext`：`context_id`, `plant_id`, `machine_id`, `material_id`, `mold_id`, `process_version`, `revision`(>=1), `observed_at_ms`, `defect`（short_shot/flash/warpage/normal）、`severity`(0..1 float)、`current: ParameterValues`、`units: dict[str,str]`、`material_viscosity: float | None`、`ambient_c: float | None`、`sample_count: int`、`measurement_ok: bool`、`synthetic: Literal[True]`。缺失的两个连续上下文只能通过显式人工补充产生新上下文版本；不猜值。
- `PARAMETER_NAMES = ('temperature','speed','pressure','cooling')`；`UNITS = {'temperature':'degC','speed':'mm/s','pressure':'MPa','cooling':'s'}`。
- 示例边界：temperature 180..240/grid1/max_delta6；speed20..100/grid5/max_delta10；pressure30..70/grid1/max_delta4；cooling8..30/grid1/max_delta3。
- `validate_context(context, now_ms) -> list[str]`：缺失、单位、范围、网格、测量、样本、未来/过期（24h）、版本等问题；无副作用。
- `check_candidate(context, raw_delta: dict[str,str], now_ms) -> dict`：验证**原始和量化后**目标/步长/方向/耦合约束，不以 clamp 修复非法原始输入。允许对原始合法 delta 精确最近格点量化，半点用 ROUND_HALF_EVEN；返回 `eligible`, `issues`, `raw_delta`, `delta`, `target`, `directions`, `context_hash`, `constraint_version`。无效输入明确 DomainError 或 eligible=False，始终 JSON 可序列化。
- 最多改变2参数；short_shot 仅允许 temperature/speed/pressure 非负；flash 仅允许 speed/pressure 非正；warpage 仅 temperature 非正/cooling 非负；normal 只允许零。至少一个非零变更才能执行型候选 eligible=True。coupling：temperature>=230 时 cooling>=15；speed>80 时 pressure<=60。模型无权修改这些 BOP 示例策略。

`fixtures.py` 必须提供：
- `demo_contexts(now_ms: int) -> dict[str, ProcessContext]` keys short_shot/flash/warpage/normal/missing/ood。ID 原创 synthetic 前缀，避免未来。正常样本 >=30 且 measurement_ok=True。process_version='synthetic-molding-v1'。
- `simulate_quality(context, parameters: dict[str,str], seed: int, sample_count: int = 60) -> dict`：真实计算合成缺陷概率/样本观测，不回传预写改善。返回 `synthetic=True`, `sample_count`, `defect_count`, `defect_rate`, `expected_defect_rate`；期望效果依赖上下文及参数，不是 BC 预测或现场因果。
- `generate_demonstrations(seed: int = 41723, groups: int = 120, per_group: int = 12) -> list[dict]`：记录 `group_id`, `context`(JSON), `delta`(四字符串)。按设备/批次血缘分组；专家由有限合法动作枚举及模拟期望目标选择，不读取未来评测集。校准和测试组与训练组不重叠。

`bc.py` 公共契约：
- `train_model(records: list[dict], *, seed: int = 41723) -> dict`：返回 `model`, `metrics`。安全 JSON 权重（不使用 pickle），神经网络真正训练；分类为参数组合 mask，回归为增量，方向从增量导出。按 group_id 严格 60/20/20 分组划分，归一化只用 train，校准只用 calibration，测试最后一次评测。返回实际训练损失/分割/校准温度/组合准确率/增量误差/无动作与简单规则对照，诚实报告。
- `validate_model(model: dict) -> None`：schema、shape、finite、hash/version/feature order 有界严格核验。
- `predict(model: dict, context: ProcessContext, now_ms: int) -> dict`：`status` 为 ok/abstain；输出 `raw_delta`, `combination`, `confidence`, `issues`, `model_hash`, `context_hash`；缺失/OOD/过期/低置信度等 abstain，不自称效果预测。模型不存在由服务明确阻断，不自动训练。
- `save_model(path: Path, model: dict)` / `load_model(path: Path)`：新路径、不覆盖、严格有界 UTF-8 JSON、无 symlink、非 pickle。训练制品只在本地数据目录，不入 Git。

## 七工具与状态边界

七个工具固定为 `get_process_context`, `retrieve_experience`, `predict_bc`, `draft_recommendation`, `check_constraints`, `request_human_input`, `submit_recommendation`。模型每回合自主选择一个可用动作，不用脚本顺序冒充 ReAct。

- 上下文/经验/BC 都由后端生成 Evidence，绑定任务、上下文 hash、版本和来源。模型不得签发 Evidence。
- 草稿只能引用实际 BC 的 raw_delta，不准模型自由编造参数；经验可支持解释，不豁免约束。
- check_constraints 冻结完整校验；submit 需该任务全部必需 evidence、当前 context/model/constraint 版本以及 exact candidate hash。
- 提交终态仅创建唯一 `pending_review` 推荐；不批准、不执行、不伪造观察。解释文字始终未审核。
- request_human_input 进入 awaiting_input；用户只能补 material_viscosity/ambient_c 以及明确说明，生成新 context revision，清空旧派生证据/候选，不清预算。
- 预算示例固定 max_calls=14、max_tools=12、max_tokens=32000、max_seconds=420、lease_seconds=900。unknown 不自动重试；明确未发送失败 usage=0，可靠已知 usage 即使无效/撤权也计费；unknown tokens=None，不等于0。
- 所有人工修改使用 owner+plant 与 acl_version 重验；CAS、幂等 request_id 绑定内容；同 key 异体拒绝。调用中取消/撤权不得提交后续工具，但不丢用量。

## 图外审批与合成设备

`pending_review -> approved/rejected -> execution_pending -> executed/unknown -> observed`。审批绑定 exact recommendation hash。执行只允许显式 confirmed=True，名字/页面醒目标记 synthetic；本地设备 revision/原参数/上下文 freshness 必须与候选相同。批准不执行；请求不等于回读；未知不盲重发。回读核验设备/请求/参数/版本，人工对账不可制造参数已生效事实。

观察必须发生在有效回读以后、有足够成熟时间及新样本；保留原始分子/分母、同口径前后差异与不确定性；不成熟不判改善。实际使用 fixtures 的计算模型，明确合成模拟，不声称现场因果效果。

## 验收和发布

真实模型、TEST、实际 BC 训练/留出评测、实际 Chrome、源码外离线安装分别验收，失败记录保留。发布前逐内容/暂存/历史审查，只提交新源码、原创合成生成器、测试、文档和依赖锁；模型权重、运行时配置、数据库、凭据、日志、浏览器 profile 都不上传。无整体开源许可、部署或 Pages。
