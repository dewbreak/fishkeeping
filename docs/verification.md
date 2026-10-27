# 验收记录

日期：2026-10-28（Asia/Shanghai）。本记录区分实际运行与只提供配置，不使用上游项目宣传指标代表本项目质量。

## 本轮：真实 DeepSeek 联调与质量修复

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

- **真实 embedding 提供商**：尚未配置独立向量服务；仅使用 MockTransport 验证格式、维度、异常与降级。DeepSeek 已完成上面的真实联调，但一次小型合成测试不等同于全面的事实准确率评测。
- **Docker 容器运行**：本机 Docker Desktop Linux 引擎管道不存在；Dockerfile/Compose 已准备并静态校验，但未成功构建或启动容器。
- **生产质量与规模**：没有正式 benchmark、压力测试、集群、TLS 网关或完整用户身份系统。首次向量缓存采用有界渐进建立；默认词法不是完整语义检索。

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
