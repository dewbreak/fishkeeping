# 验收记录

日期：2026-10-28（Asia/Shanghai）。本记录区分实际运行与只提供配置，不使用上游项目宣传指标代表本项目质量。

## 本轮：v0.5 可选择模块的 MCP 服务

新增 `fishkeeping mcp`，默认只注册七个常用记忆工具，图片两工具和视频采样可通过 `--modules` 独立组合；支持 stdio 与 Streamable HTTP。工具直接复用现有业务服务，不经过内部 HTTP，不复制 OCR/取帧/存储逻辑。官方 SDK 作为可选 `mcp` extra，锁定 2.3.0；使用当前的 `MCPServer` 接口。安装与客户端配置见 [MCP 接入](mcp.md)。

- **540 项 pytest 通过**，Ruff lint/format 通过。只新增八项实际 MCP 协议测试和两项 CLI 路由测试，聚焦模块可见性、写入/召回/来源删除、图片/OCR/视频结果；不重复原有畸形媒体或预算测试。
- **实际 stdio/HTTP 调用通过**：`scripts/smoke_mcp.py` 启动独立 stdio 子进程及回环 HTTP 服务，官方客户端完成发现九个 memory+image 工具、写入/召回/删除与原生图片块读取。预览由 800×400 缩到 512×256，全程无模型请求。
- **真实媒体通过 MCP 返回**：只开启 image/video，使用已有公开缓存的 USDA 稻米页和动画 MP4，OCR 返回 104 区域，13–17 秒返回六个带时间戳的 512×288 预览；复核标记保留、普通 JSON 无帧 Base64、没有创建记忆数据库。这里只验证本地处理与 MCP 传输，不重新宣称模型回答正确；未追加此前的付费对照。
- **原服务回归通过**：实际 HTTP + SDK + CPU OCR + 本地 MiniLM smoke 通过。新增 SDK 引入 HTTPX2 后，Starlette 测试客户端切换了响应类型；只调整三份旧测试的共用传输桥接，没有在生产 SDK 加兼容层。MCP 和现有 HTTPX 客户端可并存。
- **文档与发布**：README、MCP 客户端配置、架构、开发说明和 OpenAPI 版本同步为 v0.5；数据库 schema 不变。wheel/源码包构建与文档链接检查通过，发布内容不包含凭证、数据库或媒体缓存。

本轮 **零付费模型调用**。MCP 图片/视频由客户端模型解释，客户端自身可能计费；若配置远程 embedding，记忆检索仍沿用该提供商的计费。固定 scope 的 HTTP 入口面向可信本地调用，不新增公网多租户认证。当前测试没有覆盖所有第三方客户端 UI；工具协议、图片块和两种传输已实际验证。机器可读记录见 [MCP 验证](validation/mcp-2026-10-28.json)。

远程 CI 在推送后按实际结果补记。

## v0.4 真实照片、复杂表格与短视频（历史）

已增加冻结真实素材、复杂表格完整证据组与复核回退、带时间戳的有界视频抽帧、显式时间窗、视频 API/SDK/可靠事件、单次多图问答和持久费用预算。参考 GitHub 三个项目的固定提交与许可，没有引入 GPU 大模型或复制完整运行时。详见 [真实素材评测](media-evaluation.md)、[视频接入](video.md) 和 [调研](research/video-and-real-media.md)。

- **本地测试**：530 项 pytest 全部通过，Ruff lint/format、diff 检查通过；仅保留原有 Starlette HTTPX 弃用提示。普通测试全离线，无模型/媒体下载。新增覆盖解码超时/提前 EOF/PTS/临时文件清理、并发忙、时间窗、重复画面、表格原子预算、疑似年份表头、证据与审计一致性、付费前校验、输出拒绝和预算恢复。
- **真实 HTTP**：原有 CPU OCR+本地 MiniLM smoke 通过；新增缓存公开照片、双表和真实 MP4 的 HTTP/SDK analyze、事件、幂等、参数冲突、scope、来源级联与删除墓碑通过，零付费，[报告](validation/media-http-smoke-2026-10-28.json)。受限 Windows 默认 TEMP 不可写曾导致解码/构建失败，指定项目 `.tmp` 后同一素材通过，没有以 mock 代替。
- **真实问答**：六题、12 个方案，去重后 11 次实际请求。15,888 输入+1,216 输出 tokens，按峰值未缓存单价估算 US$0.0062256，低于 US$0.05。10 个独立输出通过格式/引用 ID 校验，最后一道视频优化输出被拒绝、未重试。玉米优化复用基线，不重复计费；调用失败也保留预算。[完整账本](validation/media-live-2026-10-28.json)。
- **保留失败**：照片 low 未降低输入 tokens；稻米 OCR 比原图增加 75.6%，两侧均错读九月列；玉米原图内容正确但缺引用；视频基线缺少完整证据，优化第二题输出失败。第一视频优化顺序经实际选中帧人工核对成立，输入减少 59.1%。不是全题通过，不使用词组命中冒充准确率。[逐项人工核验](validation/media-manual-review-2026-10-28.json)。
- **采样消融**：同窗同为六帧时 adaptive 没有优于 uniform；两条付费路线的约 59% token 降幅来自窗口/帧数/detail 的组合，不归因于单一算法。[零付费消融](validation/video-sampling-ablation-2026-10-28.json)。两个视频问题来自同一动画，不代表实拍、音频或长视频验证。
- **旧回归**：两套本地混合检索 Recall@3 仍为 0.928571/0.916667，空结果 8/8、12/12；任务时间线仍 0.90625，空结果 9/9；八张合成图片回归保留原标签、文字图 CER=0。未重跑旧付费实验或重置旧八次账本。
- **兼容与发布**：核心 schema 仍为 v2，无新增数据库迁移；手工构造 v0.3 请求哈希的默认图片事件可重放且不重新 OCR，切换 table 会冲突。视频事件只保存来源元数据，原片/帧不落库。v0.4 OpenAPI 已导出；wheel/源码包构建成功，检查没有凭证、数据库、原片、模型或缓存。文档链接检查通过。

源码提交 `7b6deced24ab37190d67e9a2bd671942359b9f9b` 的 **Ubuntu/Windows × Python 3.11/3.12 四组远程 CI 全部通过**，包括基础+dev 依赖环境的 530 项离线测试、lint/format、HTTP smoke、检索/时间线评测和构建：[v0.4 实际运行](https://github.com/dewbreak/fishkeeping/actions/runs/40567499328)。CI 未安装 vision/local 模型，也未发起付费请求；本地真实媒体验证单独记录在上文。本句为源码 CI 成功后的文档补记。

未验证 Docker 引擎运行、生产并发、远程 embedding、真实付费 ROI、跨页表格、实拍快速动作/长视频/音轨；不把本轮小样本或配置当成这些验证。

## v0.3 增量索引与 DeepSeek 低 token 图片链路（历史）

已完成独立冻结任务评测、本地增量索引、官方视觉调研、CPU OCR 与证据筛选、原生视觉/区域适配器、图片 API/SDK/可靠事件和示例。原始业务数据库、图片、凭证均未用于评测。

- **本地验证**：380 项 pytest、Ruff lint/format 通过；真实 Uvicorn HTTP + SDK + CPU OCR + 本地向量 smoke 通过；旧两套评测保持 Recall@3 0.928571/0.916667，空结果 8/8、12/12。
- **增量索引**：每批 ≤32 条/32000 字符，跨重启持久日预算与租约；新增记录、模型版本、并发领取、租约过期、删除/归档/替代/TTL 竞态、未知 schema 和事务回滚有离线测试。真实缓存模型的重开数据库和新写入补索引通过。核心 schema 仍为 v2，附加索引 worker schema 为 v1，旧 Store 可继续打开；完整回滚需停止 worker 并恢复对应备份，不通过回滚重置当天预算。
- **事件与输入边界**：图片、参数、scope 共同决定请求身份；OCR 升级后相同事件返回已保存证据。改变参数 409，删除后墓碑拒绝重放，已确认事实的来源级联通过。OCR 保存为 tool 消息，无隐式提取/付费请求。图片字节、像素、EXIF、区域、缓存 scope、低置信和畸形提供商响应有回归测试。
- **独立任务**：25 题检索 Recall@3 0.625→0.90625，9/9 严格空结果；8 场景真实最终回答子集 5/8→7/8。错误事实仍可能被召回，保留困难改写和矛盾漏召回，详见 [评测](evaluation.md)。
- **图片**：8 张原创合成图，7 张文字图本机 CER=0，目标证据短语 8/8 保留；6 次真实问答均通过冻结词组检查。截图 OCR、表格 OCR、场景 low 输入分别减少 67.2%、33.8%、55.2%。这不是实际照片/复杂文档质量保证，见 [图片报告](vision.md)。

**全部真实调用为 8 次**（6 图片对照+2批量记忆回答），4718 输入+548 输出=5266 tokens；按当日峰值未缓存价格保守估算 **US$0.002073**，不是账户扣费账单。`.tmp/vision-live-budget.json` 在每次发送前锁定并持久化预留，失败也计数，重启不重置。费用约束 ≤US$0.05、次数≤8、每次输出≤1024 均满足，已停止付费调用。[图片实际 usage](validation/images-live-2026-10-28.json) · [最终回答与总账本](validation/task-answers-live-2026-10-28.json)。

万条合成规模回归再次通过：词法候选中位数约 0.199 ms，384 维精确向量扫描约 1.004 秒；仍为内存数据库单线程诊断，非生产 p95。本阶段没有引入 ANN，也不掩盖全扫描成本。

使用 Git 中实际 v0.2 `storage.py`，验证旧数据库加上 worker 表后仍能被旧 Store 重新打开，并将升级前 SQLite 在线备份恢复为原 schema/记录；[回滚报告](validation/rollback-v3-2026-10-28.json)。wheel/源码包构建成功，OpenAPI 已按 v0.3 重新导出。图片、数据库、模型权重、缓存和凭证不进入发布包或 Git。

第一阶段提交 `6fd1593d34bbbab9056b648ddb8c9831ca51e10e` 的 Ubuntu/Windows × Python3.11/3.12 四组远程 CI 已全部通过：[运行记录](https://github.com/dewbreak/fishkeeping/actions/runs/40558312557)。最终图片功能提交 `0571f7ac138b9c7afa9678f9be2914d4a087068b` 的四组 CI 也全部通过，含 380 项离线测试、lint/format、HTTP smoke、原有与新增时间线评测、构建：[v0.3 CI](https://github.com/dewbreak/fishkeeping/actions/runs/40560450019)。这些 CI 使用基础+dev 环境，没有安装 vision/local extra 或调用付费模型。

仍未实测 Docker 引擎启动、远程 embedding、真实 ROI 付费请求、复杂照片和多页图表；不以 fake 或静态配置替代这些验证。Codex 重置授权未触发，未消耗重置卡。

## 本轮：v0.2 可靠事件、检索和本地多语言向量

按顺序完成事件持久化与独立 worker、来源完整关系和分页、FTS5 候选改造，随后接入固定版本的多语言 MiniLM CPU 模型。核心 schema v1→v2 有事务迁移和回滚测试；原始业务数据库未用于测试。

- 可靠事件：202 回执先于模型请求；租约、重启恢复、幂等、显式重试、全库 UTC 日预算、并发领取和删除竞态均有回归测试。默认每天 20 次领取、每任务最多 3 次尝试；超时或失联不自动重发。外部模型 exactly-once 不作承诺。
- 来源：测试累计 151 条来源、分页、重启、非预览来源删除与全图循环校验；响应保留 100 条预览并增加完整计数。
- 候选：测试 5001 条新噪声之后仍可找回旧记录；scope/状态/TTL/session 过滤发生在截断之前。损坏向量缓存显式降级，支持按 scope/model 重建。
- 显式索引：查询仅编码 query；索引每批最多 32 条和 32000 字符。测试 32 条各 16000 字符记录完整分批编码，尾部参与推理，超时预算后可以续建。
- 本地模型：真实下载约 252 MB 固定 revision，CPU 实际推理成功；首次暴露 HF/Xet 辅助缓存路径问题，已修复为默认跟随项目缓存目录。真实 HTTP + SDK 验证事件回执、索引、跨语言召回、来源分页和级联删除。

真实 DeepSeek 新增验收仅 **1 次 Flash 调用**，thinking disabled、输出上限 1024 tokens；14 项检查通过。记录为 **260 输入 + 73 输出 = 333 tokens，HTTP 200 / stop，耗时 1.141 秒**，见 [事件联调报告](validation/events-live-2026-10-28.json)。使用隔离数据库和合成消息；重放不新增调用。凭证及真实业务记忆未写入报告或提交。

原始 22 题中，最终本地混合检索 Recall@3 / MRR@3 从词法 **0.857143 提升至 0.928571**，8/8 空结果检查通过。额外 24 题为 **0.916667**，12/12 空结果通过。初始阈值虽然得到 100% Recall，却造成 4 个无关结果，失败报告完整保留；最终保留两项困难漏召回，详见 [评测与取舍](evaluation.md)。本地向量没有 API 费用。

万条合成记录实测：词法查询中位数 0.218 ms、完整 384 维向量扫描约 1.026 秒；这是内存数据库单线程诊断，不是生产规模承诺，见 [规模报告](validation/storage-scale-2026-10-28.json)。

本地最终验证：**312 项 pytest 测试通过**，Ruff 检查与格式检查通过；真实 HTTP smoke（含本地 CPU 模型）通过；wheel 与源码包构建成功，检查未包含凭证、数据库、模型缓存、虚拟环境和 uv 缓存；OpenAPI 快照与当前应用一致，文档相对链接检查通过。仅有此前已有的 Starlette 测试客户端 HTTPX 弃用提示。

代码提交 `2d0b10f5300012c2e54b83a475a0913834e78e2e` 的 **Ubuntu/Windows × Python 3.11/3.12 四组远程 CI 全部通过**，包括不安装可选 local extra 的基础依赖环境、312 项测试、lint/格式、真实 HTTP smoke、离线评测及构建，见 [v0.2 CI 运行记录](https://github.com/dewbreak/fishkeeping/actions/runs/40552078267)。本句是该代码提交通过后的文档补记。

## 上轮：真实 DeepSeek 联调与质量修复（历史）

北京时间 2026-10-28 00:49，使用本地 `.env` 中的 key 完成真实联调。只使用隔离数据库中的合成消息，没有读取业务记忆；key 不进入报告、日志或 Git。机器可读记录见 [deepseek-live-2026-10-28.json](validation/deepseek-live-2026-10-28.json)。

账号 `/models` 实际返回 `deepseek-flash` 和 `deepseek-v4-pro`，本轮选择较低价的 Flash，显式关闭 thinking。依据：[官方价格](https://api-docs.deepseek.com/quick_start/pricing/)、[思考模式默认行为](https://api-docs.deepseek.com/guides/thinking_mode/)。每次最多输出 1024 tokens，整轮硬限制 4 次模型调用，不自动重试。

| 实际模型调用 | HTTP / 完成状态 | 输入 / 输出 tokens | 单次耗时 |
| --- | --- | --- | --- |
| 用户事实与步骤提取 | 200 / stop | 279 / 153 | 1.562 秒 |
| 已提取事实再次巩固 | 200 / stop | 361 / 153 | 1.047 秒 |
| 原始对话巩固 | 200 / stop | 305 / 121 | 0.937 秒 |
| 根据召回记忆回答 | 200 / stop | 522 / 19 | 0.765 秒 |

合计 **1467 输入 + 446 输出 = 1913 tokens**。按当日官方峰值价格、全部输入按未命中缓存计算，费用保守估算为 **US$0.0009753**；这是 usage 推算，不是账户扣费账单。4 次调用全部正常完成，18 项端到端检查通过，包含真实提取与来源链接、事件重放不重复调用模型、重复巩固、排除助手虚构、召回及预算、基于记忆回答、作用域隔离、级联删除、删除后拒绝重放、重开数据库确认删除。

本轮修复和优化：

- 关闭 DeepSeek 默认的高强度 thinking；默认输出预算从固定 4096 调整为可配置 2048，联调脚本另收紧为 1024。
- 模型原样返回已有事实时，巩固复用该记录，避免自引用或祖先回边；独立的新来源仍合并，普通写入仍拒绝循环。
- 对话写入保存角色区间与一致性摘要。巩固只使用结构校验通过的 user 内容，避免 assistant/tool/system 文本被提升为用户事实；角色样式的正文不会改变真正角色。原始对话仍完整保留供审计。
- 提供商错误增加脱敏的稳定 `code`，区分凭证、余额、限流、超时、截断和格式错误；空白或模板 key 不触发模型调用。
- 增加显式 `--live`、调用数量与模型/输出上限约束的真实联调脚本；CI 永远不运行它。
- 新增离线检索诊断及 CI 步骤：26 条合成记忆、22 个查询，Recall@3 / MRR@3 为 **0.857143**，8/8 严格空结果检查通过，禁止记录命中为 0。保留同义词和跨语言两个失败案例，不用删题提升数字。详见 [评测说明](evaluation.md)。

本轮本地验证：**204 项 pytest 测试通过**，Ruff 检查、格式检查、真实 HTTP smoke 和离线评测通过。源码提交 `02cb1ff176777716d4b44349dc2e70aa148061a1` 的 Ubuntu/Windows × Python 3.11/3.12 四组远程 CI 也全部通过，含新增离线评测步骤，详见 [本轮 CI 运行记录](https://github.com/dewbreak/fishkeeping/actions/runs/40545834991)。此句为通过后的文档补记；下面保留首版验收历史。

## 首版已实际验证（历史）

- Windows 本地，Python 3.12.4、uv 0.11.29。
- pytest：**126 项通过**。覆盖存储持久化、隔离、TTL、去重、事实历史、派生失效、事务回滚、事件幂等与墓碑、来源删除、中文/英文/混排召回、预算、向量预热/降级、提供商输出校验、HTTP 与 SDK、并发期间来源变化。
- Ruff 静态检查通过；源码与示例统一格式化。
- 实际启动 Uvicorn 并通过 HTTP 运行 `examples/basic.py`，成功写入情景/语义记忆、获取上下文与统计，`readyz` 返回 ready。
- `scripts/smoke.py` 已实际运行通过：临时端口和隔离数据库，验证 SDK、中文混排检索、派生记忆级联删除；运行后关闭服务线程并清理临时数据库。
- 构建源码包与 wheel 成功；检查 console entry point；确认包内无 `.env`、虚拟环境、uv cache、临时数据。
- `.env.example` 可被 Settings 加载；Compose 静态配置校验通过。
- **跨平台 CI 四组全部通过**：Ubuntu/Windows × Python 3.11/3.12，包含依赖锁定安装、lint、格式、126 项测试、真实 HTTP smoke 和打包。验证的源码提交为 `c474b5810d8b007e60cc203e9b75afe264014a96`，详见 [GitHub Actions 实际运行记录](https://github.com/dewbreak/fishkeeping/actions/runs/40541583849)。本条验收记录为该源码提交通过后的文档补记。

关键依赖锁定在 `uv.lock`。本次环境：FastAPI 0.162.0、Starlette 1.10.0、HTTPX 0.28.1、Pydantic 2.14.1、Uvicorn 0.54.0、pytest 9.1.1。测试过程中 Starlette 对其测试客户端使用 HTTPX 给出一条迁移弃用提示，不影响本次通过结果；后续升级时应评估测试传输的迁移。

## 未实测的范围

- **真实远程 embedding 提供商**：远程适配器仅使用 MockTransport 验证。本地 MiniLM 已完成上面的真实 CPU 和 HTTP 验收；两者不能相互替代。小型合成测试不等同于全面的事实准确率评测。
- **Docker 容器运行**：本机 Docker Desktop Linux 引擎管道不存在；Dockerfile/Compose 已准备并静态校验，但未成功构建或启动容器。
- **生产质量与规模**：没有正式 benchmark、并发压力测试、集群、TLS 网关或完整用户身份系统。向量索引需显式建立，检索仍是全量精确扫描；多语言模型也有保留的漏召回。

## 复验命令

```text
uv sync --extra dev --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -q
uv run python scripts/smoke.py
uv run python scripts/evaluate.py --output .tmp/evaluation.json
uv build
```

受限 Windows Temp/cache 的处理方式见 [运行维护](operations.md)。验收环境使用项目内 `.tmp` 作为 pytest basetemp，测试不读取或修改真实用户数据库。

真实联调需要单独显式执行 `uv run python scripts/live_deepseek.py --live --report .tmp/deepseek-live.json`。它会使用本地 key 发起最多 4 次付费请求；不要放进普通测试循环。报告只保存检查状态、耗时、模型名和 token usage，不保存消息、模型文本或 key。

本轮可靠队列的单调用复验：`uv run python scripts/live_events.py --live --report .tmp/events-live.json`，硬限制最多 1 次付费请求。真实本地向量的复验命令见 [评测文档](evaluation.md)，需保留 `--extra local`，不使用付费 embedding 服务。
