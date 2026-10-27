# Fishkeeping

受人类记忆机制启发的独立 agent 记忆服务，提供 REST API 和 Python 客户端，可接入自己的 DeepSeek 对话循环。Python + FastAPI + SQLite，本地持久化，核心能力无需大模型 key。

## 已实现

- 工作、情景、语义、程序性四类记忆；会话工作记忆默认一小时过期。
- SQLite FTS5 全范围候选召回，可选本地多语言 CPU 向量；返回相关度与保留强度。
- 时间衰减、显式复习、重要性排序、字符预算内上下文，以及默认预览的归档维护。
- `fact_key` 事实版本、来源完整关系与分页、旧派生摘要失效；删除来源时级联删除派生记忆。
- 可靠事件摄入、持久化后台提取、幂等重放、租约和每日调用上限；可选 DeepSeek JSON 提取与巩固。
- `namespace / user_id / agent_id` 隔离、绑定 namespace 的 API key、OpenAPI 与类型化客户端。

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

首次下载约 252 MB，后续使用本地缓存。查询只编码当前问题，不会顺带构建文档索引；新记忆需要再次运行 `index`。索引不完整时返回 `hybrid_partial` 和覆盖率警告。远程 embedding 仍可独立配置；不要把聊天模型地址用于 embedding。记忆上下文应作为不可信参考资料，不作为系统指令。

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

```text
uv run --locked pytest -q
uv run --locked ruff check .
uv build
```

v0.2 本地验收：312 项测试通过，真实后台提取新增 1 次 Flash 调用、14 项检查通过；本地多语言模型和真实 HTTP 联调通过。原始诊断集召回率由 85.7% 提升至 92.9%，空结果检查全部通过，详见验收与评测记录。普通 CI 仅运行离线测试、HTTP smoke 与检索诊断，不调用付费模型、不下载权重。默认服务未开启认证，仅用于受信任本机；业务网关仍需完成用户授权。
