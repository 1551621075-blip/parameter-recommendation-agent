# 依赖、来源与发布边界

## 工程身份与授权边界

本工程发行名为 `parameter-recommendation-agent`，Python 包为 `parameter_agent`，入口为 `parameter-recommendation = parameter_agent.cli:main`。业务示例为原创合成工艺上下文；数值建议来自本地训练的 behavioral cloning（BC），LangGraph 编排受后端校验的工具调用。本工程不继承其他工程的模型、安装或现场效果验收结论。

本文不新增整体开源授权，也不把依赖许可解释成对本工程、数据或量化权重的再分发授权。公开可下载不等于所有用途均获许可。以下是公开上游身份和许可声明摘要，不是法律意见、完整 SBOM、逐文件许可证审计或供应链安全证明；发行者须核对所选版本的 LICENSE、NOTICE、二进制附带声明与模型分发条款。

## Python、锁文件与原生平台

- 声明 Python 范围为 `>=3.12,<3.13`；`uv.lock` 的解析范围为 `==3.12.*`。构建后端声明 `hatchling>=1.27`，Hatchling 不在当前应用/dev 锁条目中，不能冒充已锁定的构建环境。
- 下表记录**当前提交候选的锁定版本**，不是联网查询的“最新上游版本”。当前 `uv.lock` 有 58 个 package 条目，包含工程自身、开发工具及平台条件分支；不是某台机器的已安装包数，也不是新 venv 安装成功证据。
- lock 的 registry 是公开阿里云 PyPI 镜像；候选发行物 URL 与 SHA-256 见锁文件。哈希约束字节一致性，不证明来源可信、许可证正确或构建可复现。校验脚本不自动换源或下载。
- 本工程 wheel 要求 `py3-none-any`，但**这不表示整个依赖闭包都是纯 Python**。NumPy、pydantic-core、orjson、ormsgpack、xxhash、uuid-utils 等可能需要对应 Python/OS/架构的原生 wheel。ManagedLocalProvider 另需宿主兼容的 llama.cpp 可执行文件和动态库。
- 托管运行时及发行安装探针使用 POSIX 文件/进程能力；`fcntl`、`O_NOFOLLOW`、`pass_fds` 与复制的 `venv/bin/python` 不是 Windows 原生兼容承诺。锁内出现 Windows 或 WebAssembly 发行物不代表本应用在这些平台通过验收。
- 前端只有随 wheel 分发的 `templates/index.html`、`static/app.js`、`static/style.css`；不需要 CDN、npm 构建、Java 或 JVM。

主要直接依赖、图运行相关传递依赖和开发工具如下。版本来自 `uv.lock`；许可依据公开原始依赖说明及本地对应发行物的许可元数据，未联网复核全文。

| 依赖 | 当前锁定版本 | 用途 | 上游许可声明 |
| --- | --- | --- | --- |
| FastAPI | 0.162.0 | 本机 HTTP API | MIT |
| HTTPX | 0.28.1 | 数字回环模型协议与 TestClient | BSD-3-Clause |
| LangGraph | 1.2.5 | 实际图编排；直接依赖固定为 `==1.2.5` | MIT |
| langchain-core | 1.7.1 | 图框架的运行接口 | MIT |
| langchain-protocol | 0.0.20 | 图依赖的协议类型 | MIT |
| langgraph-checkpoint | 4.3.1 | LangGraph 传递依赖 | MIT |
| langgraph-prebuilt | 1.1.1 | LangGraph 传递依赖 | MIT |
| langgraph-sdk | 0.4.10 | LangGraph 传递依赖；不等于启用托管服务 | MIT |
| langsmith | 0.14.14 | 框架传递依赖及本地禁用 tracing 的上下文 | MIT |
| NumPy | 2.5.4 | BC 数值训练、预测与合成数据 | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` |
| Pydantic | 2.14.1 | 严格数据模型 | MIT |
| Typer | 0.27.3 | CLI | MIT |
| Uvicorn | 0.54.0 | 本机 ASGI | BSD-3-Clause |
| opentelemetry-api | 1.45.1 | FastAPI 的传递依赖；不等于启用 exporter | Apache-2.0 |
| pytest | 9.2.0 | 开发测试 | MIT |
| Ruff | 0.17.3 | 开发静态检查 | MIT |
| Hatchling | 未锁定；构建声明 `>=1.27` | 构建后端 | MIT |

NumPy 的复合许可不可缩写为仅 BSD；其他传递依赖也不能一概标为 MIT。完整闭包以 `uv.lock` 与实际选定发行物为准。公开上游参考包括 [LangGraph](https://github.com/langchain-ai/langgraph)、[LangChain](https://github.com/langchain-ai/langchain)、[LangSmith SDK](https://github.com/langchain-ai/langsmith-sdk)、[NumPy](https://github.com/numpy/numpy) 和 [OpenTelemetry Python](https://github.com/open-telemetry/opentelemetry-python)。SDK 的源码许可不替代其远端商业服务条款。

### 不启用云服务或远端 tracing

LangGraph 是实际依赖，不应套用其他工程“没有 LangGraph”的说明。当前 `AgentEngine` 在 `tracing_context(enabled=False)` 内编译并调用图，显式传入 `callbacks: []`；应用不配置 LangSmith 云端 client、远端 checkpoint 或 OpenTelemetry exporter。依赖闭包包含相关 SDK/API，不等于应用会使用其云服务。不要注入 tracing 回调、自动埋点或凭据；本说明不是对任意宿主插件、恶意依赖或自行修改代码的网络沙箱保证。

模型传输由本工程 HTTPX adapter 完成，只接受数字回环 HTTP，不信任环境代理、不跟随重定向、不自动重试、不转云端，也不会在失败后改用 TEST provider。具体生命周期和使用限制见 [本地运行契约](runtime.md)。

## 外部公开模型与引擎

以下是从原始公开资料继承的**目标身份参考**，不是本轮重新下载、检查权重或成功运行的声明。模型、llama.cpp、动态库、配置及本机路径均不进入应用 wheel/sdist；运行操作者须在仓库外独立审阅并提供。

| 组件 | 公开来源 | 目标身份 / 许可边界 |
| --- | --- | --- |
| llama.cpp | https://github.com/ggml-org/llama.cpp | `b6500`，提交 `a7a98e0fffed794396b3fbad4dcdbbc184963645`；MIT |
| Qwen3-4B-Instruct-2507 | https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 | 原模型标示 Apache-2.0；目标量化 Q4_K_M |
| Qwen GGUF 发行物 | https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF | revision `0b0406b39725d752255ffeb48f102c66f98e14aa`；转换/量化分发条款须另核对 |

目标文件 `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`，参考大小 `2,497,281,120` 字节，公开参考 SHA-256 为：

```text
3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597
```

源代码提交哈希不能代替本机构建 binary/library 的 SHA-256。托管 provider 验证操作者配置的实际文件哈希，但不自动判定其符合上述目标 revision 或许可。只调用 generation；兼容旧三角色配置不代表读取、加载或调用 BGE embedding/rerank。BC 是本项目另行训练的数值模型，不是上述语言模型，也不是 BGE 检索链。

## 冻结后的发布与离线安装命令

以下命令在精确 Git 项目根运行，前提是负责人已经冻结源码并构建、审阅当前版本的产物。它们是操作契约，**不是本次执行记录**；脚本没有 build、上传或联网补依赖功能。

```bash
.venv/bin/python scripts/check_release.py
.venv/bin/python scripts/check_release.py \
  --archive dist/parameter_recommendation_agent-0.1.0-py3-none-any.whl \
  --archive dist/parameter_recommendation_agent-0.1.0.tar.gz
.venv/bin/python scripts/check_install.py \
  --wheel dist/parameter_recommendation_agent-0.1.0-py3-none-any.whl \
  --wheelhouse "$PARAMETER_WHEELHOUSE" \
  --case short_shot
```

`PARAMETER_WHEELHOUSE` 只是 shell 展开占位变量，须由操作者显式设置为源码目录外、已审阅的平坦依赖 wheel 目录；不是自动发现的配置。不得指向 runtime、模型、生产资料或含无关文件的目录。必须事先具备本机 Python/平台适用的完整依赖闭包。安装器只从该目录离线解析二进制 wheel，**不会自行按 `uv.lock` 约束依赖版本**；要保持锁版本，负责人应只提供经锁文件 SHA-256 核对的相应发行物。

### 候选、归档及安装证据的边界

- 源文件集合动态取自精确 Git 根的 tracked 与非 ignored untracked 候选，无固定文件数清单。force-added ignored 文件拒绝且不读；不扫描父目录、环境或私有 runtime。
- 白名单只有有限根文件、`src/parameter_agent` 的平层 Python 模块/三项资源、受限 docs/tests/scripts 路径；最多 2000 个文件，单文件 2 MiB，累计 32 MiB。拒绝不安全路径、链接、特殊文件、模型、数据库、日志及配置载荷。
- 同样扫描脚本、测试及文档的秘密/个人路径/外部 Web 资源模式，只有规则名而没有命中值输出。无测试字面量豁免；这仍是有限启发式，不是完整秘密、数据权利或许可证审计。
- wheel 核对发行身份、唯一 console entry、必需模块与三项资源、RECORD 精确覆盖及逐项 SHA-256/大小，并要求 package 字节与候选源码相等。Requires-Python 只归一逗号子句顺序和首尾空白，不改变操作符或放宽边界。
- sdist 在内存中受限读取，不解压；拒绝链接、特殊成员、重复/大小写冲突和超限数据。仅除去生成的 `PKG-INFO` 后，集合与字节必须恰好等于源码候选；`PKG-INFO` 也扫描并核对身份。新增或修改本文后必须重新构建相应 sdist。
- 安装 gate 使用源码外空临时目录、独占创建的 wheel 快照和 manifest、复制的全新 venv、`python -I`、清洁环境、`--no-index`、`--only-binary=:all:` 与 `pip check`。安装/探针禁止意外网络与子进程；不为不可信依赖提供恶意代码沙箱。
- 探针校验真实 installed distribution、文件哈希、模块、安装器生成的 console script、默认 `short_shot`/显式 public case、TestClient `/health`、首页与 JS/CSS。fixture 与安装包自身的 `demo_contexts` 在固定时间下比较，禁止读源树、启动模型或请求推理。

源码测试、小型 ZIP/TAR 测试夹具和 mocked install 测试不能替代冻结后真实 source-free wheel 安装。即使完成新 venv 安装，也不是新 OS/宿主验收，不是 Qwen 推理或工艺效果证明。真实安装与最终发行验证的独立执行结果见 [验收说明](validation.md)，不沿用其他工程的 PASS。
