# HTTP 与 CLI 接口

创建 `create_app(data_root: Path, provider, *, clock=None)`；内部 Store/RecommendationService/ActionService/AgentEngine，clock传各服务，仅用于可注入测试。不得seed demo或用户。默认无OpenAPI/docs，严格本机Host、mutation Origin与Host同源、Bearer权限、no-store/CSP/nosniff；有界body1MiB、duplicate/nonfinite/严格schema拒绝。不信代理host、不允许任意CORS。run/cancel跨线程执行时支持cancel竞争，不持有数据库锁调用模型。

路由JSON返回统一Store对象（metadata+payload），列表返回 `{'items':[...]}`：

- GET `/health` -> {'status':'ok','provider':fingerprint,'synthetic_only':True}，不推理。
- GET `/` `/static/app.js` `/static/style.css`：本地无CDN。
- GET `/api/me`：user_id,role,plants,acl_version。
- GET/POST `/api/contexts`：POST ContextRequest。
- GET `/api/models`：本owner可见模型metadata+metrics摘要，**不返回完整权重**；训练/注册仅CLI离线，HTTP无路径/上传模型接口。
- POST `/api/tasks` CreateTaskRequest；GET `/api/tasks`；GET `/api/tasks/{id}`。
- POST `/api/tasks/{id}/run` body必须{}；返回最终或停止task，不后台伪成功。
- POST `/api/tasks/{id}/cancel` ActionRequest；`/supplement` SupplementRequest；`/reconcile` ReconcileRequest。
- GET `/api/recommendations`、GET `/api/recommendations/{id}`。
- POST `/api/recommendations/{id}/review` ReviewRequest；`/request-execution` ExecuteRequest；`/observe` ActionRequest（seed/sample由服务器确定，不允许用户刷seed）。
- GET `/api/executions`；GET `/api/executions/{id}`；POST `/api/executions/{id}/apply` ActionRequest；`/readback` ActionRequest。apply命名和页面醒目标记“仅合成设备”，拒绝API fault参数。
- GET `/api/audit` -> {'items':...}；owner+plant范围，不含token。

身份从Authorization Bearer头取，不允许URL/query/body token/role。模型未审核解释与权威候选/检查分区；所有模型文本仅textContent不innerHTML。

CLI entrypoint `parameter-recommendation = parameter_agent.cli:main`：
- `init DATA` 新目录，随机admin token只显一次/hash存库，demo_contexts导入；**不训练、不推荐、不批准、不执行**。parent必须已存在，不覆盖。
- `train DATA` 隐藏prompt输入token、admin/reviewer，实际generate_demonstrations/train_model/register_model到synthetic-plant；保存模型与评测在DATA内（不Git），打印真实metrics，不伪成功。产物json必须安全原子无覆盖。
- `fixture --case short_shot` 输出原创JSON，case名单固定。
- `serve DATA --test-mode` OR `--runtime-config ABSOLUTE_FILE` OR `--model-url NUMERIC_LOOPBACK_ROOT`，exact one；host127.0.0.1、port8005；自定义端口1..65535可用，不支持公网host。不把token放args/env。managed模式仅管理自有进程。

工作台最少操作：登录/退出；上下文列表与四参数单位/缺陷/质量；BC模型状态及合成评测摘要；创建/显式run/cancel/补缺；过程/经验/BC证据；候选raw delta/量化delta/target/方向/硬约束；提交后独立review；execution request/apply/readback；成熟观察；审计。浏览器token只内存credentialsomit/cache:no-store，logout abort+epoch防迟到恢复，刷新丢token。所有幂等action_id通过crypto.randomUUID生成。界面不用raw JSON大dump取代可读工作台。
