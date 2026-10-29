# 参数推荐 Agent

原创、可本地运行的工业调参参考工程：**实际训练的 BC 行为克隆模型 + 七工具 ReAct + 独立硬约束 + 人工审阅 + 合成设备执行与回读**。提供中文浏览器工作台、HTTP API、CLI、原创合成数据和回归测试。

不是原生产源码；不读取内部模型、客户数据或生产接口。所有设备行为都是本地合成模拟，不能直接用于现场自动调参。

## 解决什么问题

以原创的合成注塑场景为例：发现短射、飞边或翘曲后，需要回答“调整哪些参数、方向是什么、调整多少”，同时避免把不合法建议直接写入设备。

本工程分开回答四个问题：

1. **专家通常怎样调？** 生成合成专家示范，实际训练 NumPy MLP。组合分类头选择参数组合，四个回归头给出增量；独立校准集校准置信度，分组留出集评测模仿能力。
2. **这次允许这样调吗？** 用 Decimal 校验原始目标与量化目标的范围、网格、最大幅度、缺陷方向、最多两参数组合及耦合约束；缺失、过期、分布外和低置信度输入可以停止推荐。
3. **证据是否足够送审？** 本地语言模型逐轮从当前可用工具中选择一步。后端执行工具、绑定来源、冻结候选，只创建 `pending_review`，不自动批准。
4. **实际发生了什么？** 人工审阅、执行请求、合成设备应用、独立回读和成熟后的合成效果观察各自留痕；未知执行结果不能盲目重发。

**BC 模仿动作，不预测因果改善效果；推荐不等于批准，批准不等于执行，应用不等于回读，模拟观察不等于现场验证。**

## 链路与界面

```text
原创合成工艺上下文 ──→ 七工具 Single-Agent ReAct
                              │
      经验检索 + 实际 BC 推理 ──┤
                              ↓
                    草稿 → 独立硬约束检查
                              ↓
                       冻结候选与证据
                              ↓
                       pending_review
                              │ 图外独立人工动作
                 审阅 → 执行请求 → 合成设备应用
                              ↓
                    独立回读 → 成熟后合成观察
```

实际 LangGraph 是 `agent → guard → tools → agent` 三个节点。七个工具为：

`get_process_context`、`retrieve_experience`、`predict_bc`、`draft_recommendation`、`check_constraints`、`request_human_input`、`submit_recommendation`。

语言模型不能自由写参数、替换模型权重、伪造缺失测量值、豁免约束、批准或操作设备。Heavy Harness 落在来源版本校验、工具白名单、调用日志、预算、租约/CAS、幂等、未知结果对账和权限上，不靠增加空节点。

工作台展示原始参数与单位、BC 指标与增量、量化目标、约束、工具证据、未审核解释、审阅/执行状态和审计。无 CDN；访问令牌只放浏览器内存，刷新或退出后需重新输入。

## 快速启动

需要 **Python 3.12** 和 `uv`。已验证环境为 macOS / Apple Silicon；受管模型适配器使用 POSIX 文件锁，不声称支持 Windows。

```bash
uv sync --locked

# DATA 必须是全新的目录，父目录先存在；建议放在仓库外。
DATA="$(dirname "$PWD")/parameter-demo-data"
uv run --no-sync parameter-recommendation init "$DATA"

# init 显示一次随机令牌，请妥善保存；train 通过隐藏输入读取它。
uv run --no-sync parameter-recommendation train "$DATA"

# 先验证业务流程。TEST 是明确标记的确定性工具选择器，不是大模型。
uv run --no-sync parameter-recommendation serve "$DATA" --test-mode
```

浏览器打开 **http://127.0.0.1:8005/**，输入令牌。选择短射场景，依次点击创建任务、运行、填写审阅决定与说明、确认执行请求、确认合成应用、独立回读。普通服务中需在回读至少 60 秒后另行点击观察；不会自动批准、执行或观察。

`init` 创建原创示例，不训练 BC；`train` 才实际训练、评测并注册安全 JSON 权重及模型绑定指标。没有默认令牌、预先批准或隐藏云端后备。数据过期后不能靠补充字段刷新时间戳，应导入新的真实时间对应的合成上下文。

### 使用真实本地语言模型

可替换上面的 `serve` 命令，三种模式必须且只能选一种：

```bash
# 管理由你显式提供的本机配置；配置、模型、运行日志留在仓库外。
uv run --no-sync parameter-recommendation serve "$DATA" \
  --runtime-config /absolute/outside/repository/local-runtime.json

# 或连接你已启动的本机 OpenAI-compatible 服务（数字回环 HTTP）。
uv run --no-sync parameter-recommendation serve "$DATA" \
  --model-url http://127.0.0.1:8080
```

受管配置格式、模型来源、运行时限制与恢复说明见 [本地模型说明](docs/runtime.md)。此工程不提交模型，不自动下载或静默切换云模型。健康检查不进行模型推理。

## BC 与安全边界

- 11 维输入，48/32 隐层，11 类组合分类与 4 维增量回归；安全 JSON 权重，不使用 pickle。
- 训练/校准/测试按组分割为 60%/20%/20%；归一化仅拟合训练集，温度仅使用校准集，测试不反哺训练。
- 温度、速度、压力、冷却时间通过严格十进制字符串输入。非法原始建议直接拒绝，不能先截断再称合法。
- 对账保留已知用量；未知不当作零；过期 worker 的迟到响应不能复活已取消任务或已对账动作。
- 权限按 owner + plant + 角色重新检查。**独立审阅指独立的显式人工动作，不是强制双人复核**：当前隔离策略允许具有审阅角色的对象所有者审阅，不允许另一用户越权读取。不是企业审批流替代品。
- 证据哈希绑定内容与持久化来源，**不是密码学签名**。模型解释属于未审核文本，不能覆盖结构化事实。
- SQLite/本地文件适合此参考工程，不宣称分布式生产执行或硬实时控制能力。

## 测试与验收

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check .

# 输出目录必须全新且位于仓库外；此验收明确推进合成时钟。
uv run --no-sync python scripts/run_acceptance.py \
  --test-provider --data-root /absolute/outside/repository/new-test-run

# 真模型验收，与 TEST 结果独立。
uv run --no-sync python scripts/run_acceptance.py \
  --runtime-config /absolute/outside/repository/local-runtime.json \
  --case short_shot --data-root /absolute/outside/repository/new-real-run

# 无副作用默认模式；显式 --run-chrome 才启动独立 Chrome profile。
node scripts/browser_smoke.mjs --self-test
node scripts/browser_smoke.mjs --run-chrome \
  --artifacts-parent /absolute/existing/outside/repository
```

浏览器验收要求已安装 Chrome 与 Node 24，不下载浏览器、不读取用户 profile。它操作真实 DOM/API、实际训练的 BC，但工具编排明确为 TEST；观察成熟使用显式合成时钟，不冒充真实等待或物理采样。

已完成的独立验收：

| 项目 | 结果 |
|---|---|
| 完整 pytest 回归 | 1624 通过，无失败或跳过 |
| Ruff / JS 语法 / 锁文件检查 | 通过 |
| 实际 BC 留出评测 | 组合准确率 93.75%，门控覆盖率 70.14%；不是改善成功率 |
| TEST 业务集成 | 六场景通过，包括缺失补问与正常/OOD 拒绝推荐 |
| 真实本地 Qwen | 最终短射场景 6 次调用 / 6 次工具 / 8402 tokens，350.893 秒 |
| 实际 Chrome | 最终 20 项通过 / 0 失败；工具选择为 TEST |
| 打包与安装 | wheel/sdist 与源码核对通过；源码外全新 venv 离线安装、入口/API/资源探针通过 |

详见 [验收说明](docs/validation.md)。保留首次真实模型失败记录；没有将局部测试相加、TEST 当作 LLM、模拟观察当作现场验证。新 venv 仍共享宿主系统，不是新机器验收。

## 文档

- [架构与状态闭环](docs/architecture.md)
- [CLI / API / 工作台使用](docs/usage.md)
- [依赖、模型来源与许可边界](docs/dependencies.md)
- [本地模型配置与恢复](docs/runtime.md)
- [实施契约](docs/implementation-contract.md)
- [存储契约](docs/storage-contract.md)
- [服务契约](docs/service-contract.md)
- [API 契约](docs/api-contract.md)

公开依赖和模型各自遵循上游许可；未替整个原创工程擅自授予 MIT/Apache 开源许可。不要提交数据目录、令牌、运行配置、模型权重、数据库、验收原始记录或现场材料。
