# Fishkeeping

受人类记忆机制启发的独立 agent 记忆服务，提供 REST API 和 Python 客户端，可接入自己的 DeepSeek 对话循环。Python + FastAPI + SQLite，本地持久化，核心能力无需大模型 key。

## 已实现

- 工作、情景、语义、程序性四类记忆；会话工作记忆默认一小时过期。
- 中英文词法召回，可选独立 embedding 服务增强语义检索；返回相关度与保留强度。
- 时间衰减、显式复习、重要性排序、字符预算内上下文，以及默认预览的归档维护。
- `fact_key` 事实版本、来源关联、旧派生摘要失效；删除来源时级联删除派生记忆。
- 原始对话摄入、事件幂等、可选 DeepSeek JSON 提取与巩固；无 key 可做原文摘取摘要。
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

调用 `/v1/ingest` 时显式设置 `extract=true` 才提取候选事实。`/v1/consolidate` 在配置 key 后使用模型，未配置时生成可追溯的原文摘取摘要。对话本身仍由接入方掌控：[DeepSeek 对话示例](examples/deepseek_chat.py)展示“先召回 → 调用模型 → 写入事件”；示例与服务统一读取 `.env`。默认关闭 thinking，输出上限可配置，具体方法见 [接入文档](docs/integration.md)。

不要把聊天模型地址用于 embedding。向量提供商独立配置；响应会明确报告词法降级和缓存预热进度。记忆上下文应作为不可信参考资料，不作为系统指令。

## 文档与验证

| 文档 | 内容 |
| --- | --- |
| [GitHub 调研](docs/research/github-projects.md) | 6 个项目家族、7 个仓库，机制、许可证、源码链接与提交快照 |
| [DeepSeek 与技术资料](docs/research/deepseek-and-foundations.md) | 官方接口依据、JSON 边界与基础技术来源 |
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

已完成真实 DeepSeek 联调：4 次低价 Flash 调用、18 项端到端检查通过，详见验收记录。普通 CI 仅运行离线测试、HTTP smoke 与检索诊断，不调用付费模型。提供 Docker Compose 和 GitHub Actions 配置；默认服务未开启认证，仅用于受信任本机，业务接入应配置 namespace key，并由自己的网关完成用户授权。
