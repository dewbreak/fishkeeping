# Fishkeeping

受人类记忆机制启发的独立 agent 记忆服务，提供 REST API 和 Python 客户端，可接入自己的 DeepSeek 对话循环。Python + FastAPI + SQLite，本地持久化，核心能力无需大模型 key。

## 已实现

- 工作、情景、语义、程序性四类记忆；会话工作记忆默认一小时过期。
- SQLite FTS5 全范围候选召回，可选本地多语言 CPU 向量；返回相关度与保留强度。
- 时间衰减、显式复习、重要性排序、字符预算内上下文，以及默认预览的归档维护。
- `fact_key` 事实版本、来源完整关系与分页、旧派生摘要失效；删除来源时级联删除派生记忆。
- 可靠事件摄入、持久化后台提取、幂等重放、租约和每日调用上限；可选 DeepSeek JSON 提取与巩固。
- `namespace / user_id / agent_id` 隔离、绑定 namespace 的 API key、OpenAPI 与类型化客户端。
- 可恢复的本地增量索引 worker：自动发现新记忆，持久租约与全库每日计算预算。
- 图片证据 API/SDK：CPU OCR、问题相关区域、来源哈希与坐标；显式 DeepSeek 原生视觉或紧凑 OCR 问答。
- 复杂表格的整组证据保留与保守回退；短视频本地抽帧、显式时间窗、时间戳来源和单次多图问答。

“仿照人类记忆”在这里是工程设计方向，不代表完整模拟人脑。默认词法模式不保证同义词或跨语言理解；首版面向单机可信后端。

## 五分钟启动

需要 Python 3.11+，推荐 3.12；在项目根目录执行：

```powershell
Copy-Item .env.example .env
uv sync --extra dev --locked
uv run fishkeeping serve
```

macOS / Linux 将第一行换成 `cp .env.example .env`。服务默认监听 `127.0.0.1:8765`，打开 [交互 API 文档](http://127.0.0.1:8765/docs)。没有 uv 时的安装方式见 [运行维护](docs/operations.md)。

在另一终端运行：

```text
uv run python examples/basic.py
```

或者直接使用客户端：

```python
from fishkeeping.client import MemoryClient
from fishkeeping.models import Scope

with MemoryClient("http://127.0.0.1:8765", Scope(user_id="alice")) as memory:
    memory.remember(
        "Alice 喜欢 Python 示例和中文解释。",
        kind="semantic",
        fact_key="preferences",
    )
    recalled = memory.context("Alice 喜欢什么语言的示例？", max_chars=3000)
    print(recalled.context)
```

## 接入 DeepSeek

在 `.env` 中设置 `FISHKEEPING_DEEPSEEK_API_KEY`，重启服务。模型名通过 `FISHKEEPING_DEEPSEEK_MODEL` 配置，调研当日官方示例为 `deepseek-flash`；以账号实际支持的模型为准。[官方说明](https://api-docs.deepseek.com/)

推荐通过 `/v1/events` 或 SDK `enqueue()` 写入事件，收到 `202` 即表示原始事件与任务已经落库。另一个终端运行 `uv run fishkeeping worker`，后台才会执行事实提取；默认每天全库最多领取 20 次模型任务，每个任务最多尝试 3 次。超时和失联不自动重发，避免重复付费。兼容的 `/v1/ingest` 保留同步原子摄入语义。`/v1/consolidate` 是独立显式模型操作，不受 worker 日预算约束。

[DeepSeek 对话示例](examples/deepseek_chat.py)展示“先召回 → 调用模型 → 可靠写入事件”；示例与服务统一读取 `.env`。默认关闭 thinking，输出上限可配置，具体方法见 [接入文档](docs/integration.md)。

多语言向量选用固定版本的 MiniLM ONNX 模型，本地 CPU 推理，没有 embedding API 费用。在 `.env` 设置 `FISHKEEPING_EMBEDDING_BACKEND=local`，安装并显式构建索引：

```text
uv sync --extra dev --extra local --locked
uv run --extra local fishkeeping index --user-id alice
uv run --extra local fishkeeping serve
```

首次下载约 252 MB，后续使用本地缓存。查询只编码当前问题；运行 `uv run --extra local fishkeeping index-worker --user-id alice` 可自动补齐后续新记忆（默认最多 20 批、全库 UTC 每日 1000 条尝试）。进程需自行启动并在达到上限后调度重启，详见 [增量索引](docs/research/incremental-index.md)。索引不完整时返回 `hybrid_partial` 和覆盖率警告。远程 embedding 仍可独立配置；不要把聊天模型地址用于 embedding。记忆上下文是参考资料，不是系统指令。

## DeepSeek 低 token 识图

2026-10-28 官方 `deepseek-flash` 已支持真正的 `image_url` 图片输入。文字截图/简单表格可先本地 OCR 并筛选问题证据；非文字场景或细节不足时显式使用原生视觉 `low` / 原图 / 最多两个区域。不会把 Base64 当普通文本发送，也不会自动付费回退。

```text
uv sync --extra dev --extra local --extra vision --locked
uv run --extra local --extra vision fishkeeping serve
uv run --extra local --extra vision python examples/image_question.py screenshot.png "目标温度是多少？"
```

示例默认只做本地 OCR 和可靠事件保存；添加 `--live` 才调用一次 DeepSeek。场景图片选择 `--mode low --live`。接口、来源和限制见 [图片接入指南](docs/vision.md)；[官方能力与许可调研](docs/research/deepseek-vision.md)。

复杂表格使用 `--layout table`，歧义时保留整页，超预算要求原图复核。视频示例同样默认零模型调用：

```text
uv run --extra local --extra vision python examples/video_question.py clip.mp4 "13到17秒发生了什么？" --start 13 --end 17
```

视频采用带时间戳的稀疏图片证据，没有音频理解；详见 [视频接入](docs/video.md) 和 [真实素材评测](docs/media-evaluation.md)。

## 文档与验证

| 文档 | 内容 |
| --- | --- |
| [GitHub 调研](docs/research/github-projects.md) | 6 个项目家族、7 个仓库，机制、许可证、源码链接与提交快照 |
| [DeepSeek 与技术资料](docs/research/deepseek-and-foundations.md) | 官方接口依据、JSON 边界与基础技术来源 |
| [多语言向量选型](docs/research/multilingual-embeddings.md) | 固定模型、许可证、CPU 推理、长文本分段与限制 |
| [架构与选型](docs/architecture.md) | 技术决策、数据流、记忆生命周期和边界 |
| [接入指南](docs/integration.md) | HTTP、SDK、认证、错误码与幂等语义 |
| [运行维护](docs/operations.md) | 本地/Docker 启动、配置、备份、恢复、维护 |
| [开发交接](docs/development.md) | 模块职责、算法、后续开发优先级 |
| [验收记录](docs/verification.md) | 实际执行的验证与未验证范围 |
| [检索评测](docs/evaluation.md) | 不调用模型的固定诊断集、指标与保留的失败案例 |
| [图片识别](docs/vision.md) | API/SDK、本地 OCR、原生视觉、费用和真实对比 |
| [视频识别](docs/video.md) | 本地有界解码、问题时间窗、帧来源、可靠事件 |
| [真实素材评测](docs/media-evaluation.md) | 真实照片、复杂表格、多镜头动画；实际 token 与失败边界 |

```text
uv run --locked pytest -q
uv run --locked ruff check .
uv build
```

v0.4 本地验收：530 项测试通过，真实媒体 HTTP、CPU OCR 与本地向量联调通过。六题真实素材诊断共 11 次 Flash 请求，峰值费用估算 US$0.0062256；两道视频题输入约减 59%，稻米表仍有列关联错误，另有一次视频输出校验失败，均保留在 [评测报告](docs/media-evaluation.md)。旧检索指标无回退，详见 [验收记录](docs/verification.md)。普通 CI 不调用付费模型、不安装 OCR/向量权重；默认服务未开启认证，仅用于受信任本机，业务网关仍需完成用户授权。
