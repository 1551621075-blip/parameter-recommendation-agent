# 参数推荐本地运行契约

## 数值模型、语言模型与人工权限分开

本工程使用原创合成工艺上下文、本地训练的 behavioral cloning（BC）数值模型和实际 LangGraph 工具编排。BC 预测、工艺硬约束、证据签发、权限与状态转换由后端决定；语言模型只能从本轮发放的菜单提出一步工具动作，不能自行写参数、签证据、批准建议或执行设备。

`submit_recommendation` 仅创建待人工审阅的建议。审阅、执行请求、合成设备 apply/readback 与观察是独立授权动作；模型的解释是供人审阅的文本，不是现场有效性、改善结果或安全认证。本应用只面向合成数据/合成执行，没有生产设备接入能力。公开模型与上游许可见 [依赖说明](dependencies.md)。

## 当前 CLI 与 HTTP 入口

下列命令假定已按负责人流程准备环境、入口命令可用。`PARAMETER_DATA` 由操作者设为**仓库外、父目录已存在的新数据目录**；`PARAMETER_RUNTIME_CONFIG` 由操作者设为仓库外明确审阅的绝对配置路径。这里只给占位变量，不给本机路径或凭据。

```bash
parameter-recommendation --help
parameter-recommendation fixture
parameter-recommendation fixture --case short_shot

# init 会创建新数据目录并只显示一次本地管理员 token，禁止录屏/重定向公开日志。
parameter-recommendation init "$PARAMETER_DATA"

# 真正训练原创合成示范上的 BC，并注册模型/指标；token 通过隐藏 prompt 输入。
parameter-recommendation train "$PARAMETER_DATA"

# 下列 serve 三种模式恰好选一种；不要同时运行。
parameter-recommendation serve "$PARAMETER_DATA" --runtime-config "$PARAMETER_RUNTIME_CONFIG"
parameter-recommendation serve "$PARAMETER_DATA" --model-url http://127.0.0.1:18080
parameter-recommendation serve "$PARAMETER_DATA" --test-mode
```

这些变量仅由 shell 展开，**应用不从环境自动发现配置或认证 token**。不要把 token 放参数、环境变量、query、文档或日志。`init` 不接受已有目录，不训练模型、不创建建议、不批准或执行。`train` 的本地训练不是语言模型调用，也不是生产数据有效性证据；角色/工厂权限在训练前与发布前核对。

`fixture` 不持久化，默认 `short_shot`，还支持 `flash`、`warpage`、`normal`、`missing`、`ood`，输出严格 `ProcessContext` JSON；时间来自当前时钟。不要把 `missing`/`ood` 当作允许补造输入或强制预测的开关。

HTTP 默认监听 `127.0.0.1:8005`，可用 `--port` 更改端口，但不能用 CLI 改为公网 host。访问 `http://127.0.0.1:8005/` 使用同源本地 HTML/JS/CSS；`GET /health` 公开返回 `status: ok`、provider 指纹与 `synthetic_only: true`。**`synthetic_only` 描述业务范围，不表示 provider 一定是 TEST**。该健康请求不加载模型、不做推理；CLI 启动输出的 `inference_checked: false` 也不得解释为运行通过。

除首页、health 和两项静态资源外，API 需要 Bearer 认证，身份/工厂范围由服务端验证。HTTP 使用数字回环 Host，拒绝 query、跨源写请求与过大/不合法 JSON；不信任代理头。当前上下文、任务、人工审阅与执行路由见 [API 契约](api-contract.md)。原始数据、token、模型回复或 runtime 配置不得复制到公开日志。

## TEST 与真实本地推理模式

| 显式模式 | 含义 | 不证明什么 |
| --- | --- | --- |
| `--test-mode` | `SyntheticProvider`，指纹前缀 `synthetic-test-`；确定性 TEST ONLY 工具选择，非 LLM | 不能证明 Qwen、CPU 引擎或真实模型行为通过 |
| `--model-url` | `LocalProvider`，指纹前缀 `local-`；访问操作者已启动的本机服务 | 不验证远端进程加载的模型字节，也不拥有其清理/缓存生命周期 |
| `--runtime-config` | `ManagedLocalProvider`，指纹前缀 `managed-local-cpu-`；配置哈希校验、CPU 子进程与作用域清理 | 构造/health 成功不等于模型加载、动作有效、建议闭环或工艺效果通过 |

模式不可混用，无默认 provider、云端 fallback 或失败后 synthetic fallback。图执行明确包在 `tracing_context(enabled=False)` 内，传 `callbacks: []`；不向 LangSmith 云端传 trace，也未配置 OpenTelemetry exporter。锁文件含 tracing/SDK 传递依赖不是启用服务的证据；不得注入外部 tracing 回调、自动埋点或凭据。

## 外部已启动的本地服务

CLI 的 `--model-url` 只接受无凭据、无 query/fragment 的数字回环 **HTTP 根 URL**，支持 IPv4 或方括号 IPv6；拒绝 `localhost`、远端 IP、HTTPS 和额外路径。provider 的底层 URL 校验也拒绝 DNS、URL 间接跳转、控制字符等。请求使用 `trust_env=False`、`follow_redirects=False`，不走环境代理、不自动重试或追加“修复”调用。

协议为 `POST /v1/chat/completions`，模型别名 `qwen3-4b-instruct-2507`，温度 0、严格 JSON schema、`max_tokens: 768`、`cache_prompt: true`。每轮只能返回恰好一个 `{tool, arguments}` JSON 对象，tool 必须属于本轮 available_tools。响应必须恰好一个完整 `stop` choice，不能带额外 function/tool call、refusal 或截断结果，且 `usage.total_tokens` 必须为有效正整数。

外部模式的 `session()` 是无资源上下文，`close()` 不停止别人管理的服务。相同别名不等于相同模型哈希，也不保证 prompt/cache 未被其他调用者共享。需要受控文件身份和进程范围时，显式选择托管模式。

## 托管 CPU 配置形状

配置须由可信本地操作者显式提供；JSON 大小不超过 64,000 字节，严格字段/类型，拒绝额外字段、重复 JSON 字段、非有限数值、递归配置或任意 shell/download 指令。下面只给**字段占位形状，不是可直接运行的配置**；尖括号值必须替换为真实审阅值，`bytes` 必须替换为正整数，不能保留占位字符串。

```json
{
  "binary": {"path": "<ABSOLUTE_BINARY_PATH>", "sha256": "<BINARY_SHA256>"},
  "libraries": [
    {"path": "<ABSOLUTE_LIBRARY_PATH_BESIDE_BINARY>", "sha256": "<LIBRARY_SHA256>"}
  ],
  "models": {
    "generation": {
      "path": "<ABSOLUTE_GENERATION_GGUF_PATH>",
      "sha256": "<GENERATION_SHA256>",
      "bytes": "<POSITIVE_INTEGER_BYTE_COUNT>"
    }
  },
  "lock_path": "<ABSOLUTE_LOCK_PATH_WITH_EXISTING_PARENT>",
  "port": 18080
}
```

| 字段 | 实际约束 |
| --- | --- |
| `binary` | `{path, sha256}`；可执行 llama-server 普通文件；SHA-256 为 64 位小写十六进制 |
| `libraries` | 1–30 个 `{path, sha256}`；全部与 binary 同目录 |
| `models` | 首选恰好 `generation`；兼容恰好 `generation`、`embedding`、`rerank` 的旧三角色集合 |
| model record | `{path, sha256, bytes}`；bytes 为真实正整数、禁止 bool/字符串代替 |
| `lock_path` | 已有父目录内的锁文件；新建时权限 0600，须单硬链接普通文件 |
| `port` | 默认 18080，严格整数 1024–65535 |

配置内所有路径须绝对、彼此不同、无 traversal/控制字符；实际打开的配置、binary、library、generation 与 lock 路径链禁止符号链接。旧 embedding/rerank 描述只做结构与路径字段校验，**不打开、不哈希、不加载、不计入运行指纹**。构造 provider 只读取明确配置与本身源码身份，不启动或哈希模型；首次合法 `decide()` 的传输准备才核验并惰性启动 generation。

公开目标为 Qwen3-4B-Instruct-2507 GGUF Q4_K_M 与 llama.cpp `b6500` CPU；模型 revision/参考哈希见依赖说明。配置校验只证明操作者选择的文件符合其声明，不能证明该选择的许可证、模型效果或其与参考构建相同。不得把源提交哈希当 binary/library 哈希。

### 原生文件与启动边界

- 运行需要原生 POSIX 环境、适配 CPU/OS 的可执行文件/动态库，以及 Python 3.12 环境。`fcntl` 与文件描述符继承是实际要求，不能把 pure-Python 应用 wheel 解读为 Windows 或任意主机可运行。
- 核验 binary 可执行、各实际使用文件为普通文件、generation 的真实字节数及 binary/libraries/generation SHA-256。身份缓存绑定设备/inode/大小/mtime/ctime；变化后重新核验，不按路径名盲目信任。
- 固定 CPU 参数为 `--n-gpu-layers 0 --device none --no-op-offload --flash-attn off`，不是 GPU 失败后的静默 CPU 回退。并行度 1、CPU/批处理线程 4、HTTP 线程 2、context 8192、batch 512、ubatch 128、seed 42、温度 0。
- 只绑定 `127.0.0.1`；关闭引擎 Web UI、slots 页面及引擎日志。启动使用参数数组，不经 shell；最小环境与临时 HOME/TMPDIR，stdin/stdout/stderr 均接空设备，不保留原始模型回复日志。
- 线程作用域租约、整次 decision 锁和非阻塞 `fcntl` OS 锁排除重叠子进程。锁 fd 传给子进程；端口占用时拒绝启动，不借用未知服务器。锁父目录、引擎路径和配置由操作者独立维护，应用不自动扫描或创建模型仓库。

### `session()`、`close()` 与 cleanup-unknown

`session()` 只包围一次图 `run` 调用，不包围持久化任务的整段生命期或等待人工补充的暂停。首个 decision 惰性启动，当前作用域内后续 decision 可复用该进程及其 prompt/context cache；退出、失败或取消走作用域清理，之后另一次 run 可以建立新作用域。直接单次 `decide()` 会自动创建自己的短作用域。一个作用域失败后不能继续复用为成功状态；不自动重试推理。

`close()` 与退出 session 不同：它永久关闭该 provider 实例，空闲时幂等释放已有资源；decision 正在执行时拒绝 close，不用异步杀进程伪装取消完成。CLI 在 `serve` 结束时调用 close。正常作用域退出并不永久关闭 adapter，因此不应在每次人工暂停后误用 close 而期待原实例继续工作。

清理先 `terminate` 后等待，必要时 `kill` 再等待，每阶段等待上限 10 秒。若不能确认回收，实例标记 `_cleanup_unknown`，之后拒绝新推理并报需操作者对账；**业务任务的 `/reconcile` 不能解除这个进程安全锁**。操作者须独立核实并处理所拥有的进程/锁，然后显式重启应用创建新 provider；不要盲目删除锁文件或只重试请求。已确认返回的调用不重复计费，自动单次作用域清理失败时仍保留刚才调用的已知 usage。

## 工具、计费与时间预算

当前工具集合为 `get_process_context`、`retrieve_experience`、`predict_bc`、`draft_recommendation`、`check_constraints`、`request_human_input`、`submit_recommendation`。这不是固定由语言模型强行逐步执行的脚本；每轮资格与菜单由后端当前状态决定。模型不可虚构 evidence_id、context_hash、candidate_hash、参数值或缺失测量。缺失测量可请求人工补充；正常、分布外或低置信度等预测弃权停止推荐，不一律转成补问。提交引用确切候选与已签发证据，仍无批准或执行权。

单次输出 `max_tokens = 768` **不是输入输出总计费量**。adapter 先读取可信 `usage.total_tokens` 再校验 envelope/content/action，因此已知消耗的无效动作也计入 token 账；不能因为 JSON/工具失败就记零。只有确认未发送的错误才报告 0；已跨 dispatch 边界但 usage/结果不确定时为 unknown，不能伪称免费或自动重试。cache 命中也不能自行从服务返回的 total_tokens 扣除输入量。

当前图的协作预算为最多 14 次模型调用、12 次工具调用、32,000 累计 tokens、420 秒累计业务耗时。预算在步骤/返回边界核对，不是硬实时 watchdog。托管 startup 名义上限 90 秒，HTTP 请求超时 180 秒；阻塞传输、模型加载和清理可使墙钟超出业务阈值。退出回收时的 kill 升级不是到业务时间即刻硬杀的保证。

响应还受请求 64,000 字节、上下文 48,000 字节、动作 8,000 字节、HTTP 响应 128,000 字节以及阶段 deadline/结构深度等约束；超限失败不应触发云端、修复调用或更换 TEST provider。

## 实际证据与范围限制

首次真实 Qwen 尝试为一次调用、373 tokens，因无参工具带额外字段而失败；该失败保留，不免除用量。修正精确参数 schema/prompt 并完成恢复边界修正后，最终冻结核心的短射复验通过：6 次实际调用、6 次工具执行、8402 tokens，350.893 秒；后续人工动作与合成执行由验收脚本另行明确调用。它不是六场景 LLM 通过或现场验证。完整回归、历次独立复验及源码外安装状态统一见 [验收说明](validation.md)，不要借用其他工程或单元测试的成功记录。

`check_release.py` 验证公开候选/显式归档边界；`check_install.py` 在真正执行时验证源码外新 venv 中的发行安装、入口、合成 fixture、模块、health 与页面资源，不读取 runtime 配置或启动模型。单元测试内的 mocked install PASS 不是实际 source-free 安装成功，health PASS 不是 Qwen PASS。source-free fresh venv 仍共享当前宿主 OS、基础 Python 与内核，不是新操作系统验收，更不能外推生产可靠性或现场工艺效果。
