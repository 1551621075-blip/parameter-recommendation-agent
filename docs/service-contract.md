# 服务、图外执行与请求接口

与 implementation-contract.md/storage-contract.md 一致。服务不启动模型、不自动训练/批准/执行。所有作用域按 owner+plant；`now_ms` 默认为当前 UTC 毫秒，可测试注入 `clock: Callable[[],int]`。

## RecommendationService（service.py）

构造 `RecommendationService(store: Store, *, clock=None)`，属性 `store`。

- `import_context(principal, context: ProcessContext, action_id: str) -> object`：kind=context，id=context.context_id，payload=JSON context。新上下文只追加，不覆写既有ID。同事务为未存在 machine 创建 kind=device、id=machine_id，payload见下。已有设备的 plant/process/current 等必须匹配，不无声复位。审核权限非必要，operator允许合成输入。idempotent。
- `register_model(principal, model:dict, plant_id:str, action_id:str) -> object`：admin/reviewer，bc.validate_model，kind=model，id由owner+plant的稳定SHA256派生为有界Identifier，payload={'model':model}。不存在创建，存在CAS更新；active只有一个，旧任务绑定的版本因此失效。idempotent。
- `get_context(principal, context_id, *, conn=None) -> object`：Store对象payload可ProcessContext严格重载。
- `get_model(principal, plant_id, *, conn=None) -> object|None`：本owner该工厂模型，不跨owner共享；缺失返回None。
- `snapshot(principal, context_id, *, conn=None) -> dict`：`context` JSON, `context_hash`=digest(context), `context_version`=object version, `model` model dict或None, `model_hash`=model['model_hash']或None, `model_content_hash`=model object content_hash或None, `model_version`或None, `experience_version`='synthetic-experience-v1', `constraint_version`='synthetic-bop-v1'。校验对象、实际模型、版本绑定；不自动忽略坏制品。
- `experience(principal, context_id) -> dict`：从代码内原创小经验库按 defect/material/process 真实检索，返回 `items`（id/text/provenance/version）、query/context_hash、experience_version；不能返回硬编码故障答案。非相关返回空。
- `supplement_context(conn, principal, context_id, *, expected_hash, fields:dict, note:str) -> object`：只 material_viscosity/ambient_c、原值必须None、严格有限适用范围、非空note；版本+1；观测时间不自动刷新；审计。仅 Agent.resume 调用，不另暴露绕过接口。
- `submit_recommendation(conn, principal, *, task_id, context_id, expected_snapshot:dict, candidate:dict, evidence:list[dict], explanation:str) -> object`：重验当前context/model/constraint版本、候选hash（hash排除candidate_hash字段）、raw_delta重新约束eligible且和冻结结果一致、签发证据hash/任务/context绑定；只创建kind=recommendation id='rec-'+task_id，status=pending_review。payload至少 task_id/context_id/context_hash/context_version/model_hash/model_version/model_content_hash/constraint_version/candidate/evidence/untrusted_model_draft/synthetic=True。候选不会被文本改写。相同task唯一，不自动批准执行。`task_id`生成限制<=60以保证rec-id长度。

每个 evidence：`evidence_id`(标识), `task_id`, `tool`, `context_hash`, `payload`(实际工具结果), `content_hash`(整个evidence除自身content_hash的digest)。候选：包含check_candidate完整返回，再添 `task_id`, `context_id`, `model_hash`, `candidate_hash`(排除自身字段的digest)。expected_snapshot仅传来源键，不需完整 model。

## ActionService（actions.py）

构造 `ActionService(service: RecommendationService, *, clock=None)`。

- `review(principal, recommendation_id, *, action_id, recommendation_hash, decision, note) -> object`：reviewer/admin；decision approved/rejected，精确Store content_hash，note非空。状态pending_review；approval payload记录 decision/hash/user/time/说明合成。批准不启动任何设备动作。
- `request_execution(principal, recommendation_id, *, action_id, recommendation_hash, confirmed: bool) -> object`：confirmed必须`is True`；status approved；hash是批准后当前完整recommendation hash；重新重验源context/model、freshness/constraints、设备version/current/context绑定。创建 kind=execution id='exec-'+recommendation_id，status=pending，payload target/before/request/owner/device expected revision/recommendation_hash/source identities。recommendation status=execution_pending，禁止重复动作；未执行设备。
- `apply_synthetic(principal, execution_id, *, action_id, fault: str|None=None) -> object`：明确合成适配器，不接任意URL/代码；fault只测试可用，不经HTTP用户传入。正常实际更新device参数/version，持久化device_receipt，execution status=applied（非executed，尚未回读）；fault='after_apply' 在真实设备状态更新/receipt事务提交后模拟ACK丢失，execution status=unknown，recommendation status=unknown。相同设备端request_id幂等，不盲重发unknown；未知用readback对账。独立事务和重启恢复必须可验证，避免仅抛异常不持久unknown。
- `readback(principal, execution_id, *, action_id) -> object`：只applied/unknown，读取真实本地device_receipt+device参数/revision，精确核验request/target/before/hash/身份；存在真实匹配receipt才execution=executed、recommendation=executed，并保存readback_hash/time。没有证据保持unknown，不接受人工直接宣称成功参数。
- `observe(principal, recommendation_id, *, action_id, seed:int=27183, sample_count:int=120) -> object`：status executed，回读后至少60秒（clock可注入）、样本>=60<=10000；未成熟明确拒绝，不判改善。验证真实receipt/设备仍相同target/source，用simulate_quality计算新before/after独立合成采样，返回分子/分母/差值、二项不确定性区间及结论（synthetic_observed_improvement/no_clear_improvement/worsened）；causal_effect_proven=False；status observed。before模拟不是已观测的原生产基线，页面明确配对模拟比较。seed由服务器固定默认，不经UI挑seed来刷改善。
- `get/list`不用另写，调用Store统一对象读取；每步事务、require/CAS、receipt与audit，幂等必须先重验授权/目标再回放。用户不能借同一个action_id换body/目标/阶段。

Device payload：machine_id,plant_id,process_version,context_id,context_hash,current(四字符串),revision(>=1),synthetic=True,updated_at_ms。device是kind=device。device_receipt是kind=device_receipt id=execution_id，payload含execution_id,target,before,device_revision,applied_at_ms,recommendation/source hash。不能把本地合成原子更新包装成对真实设备的 exactly-once 承诺。

## Agent public interface（main）

`AgentEngine(service, provider, *, clock=None, monotonic=None)`。
- `create(principal, context_id, *, action_id) -> task object`。
- `run(principal, task_id) -> task object` 同步有限ReAct循环，后台API线程可取消。
- `get(principal, task_id) -> task object`。
- `cancel(principal, task_id, *, action_id) -> task object`。
- `resume(principal, task_id, *, action_id, fields:dict, note:str) -> task object` 补缺数据但不自动run。
- `reconcile(principal, task_id, *, action_id, call_id, no_result:bool, tokens:int|None, note:str) -> task object` admin，未知无结果对账不添加回复/措施；tokensNone保守占剩额，不解除托管进程cleanup_unknown锁。
- task public payload：context_id, source_snapshot(来源键), evidence(列表), prediction, candidate, status, calls, tools, tokens, elapsed_s, error, pending_action, awaiting_fields, lease_id, lease_until_ms；UI不要依赖模型内部推理。

状态 created/running/awaiting_input/blocked/submitted/no_recommendation/cancelled。用户正文不允许传role/owner、路径、endpoint、seed或任意SQL/Python。
