# 首版验收记录

日期：2026-10-28（Asia/Shanghai）。本记录区分实际运行与只提供配置，不使用上游项目宣传指标代表本项目质量。

## 已实际验证

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

- **真实 DeepSeek / embedding 提供商**：没有使用用户凭证发起付费请求。使用 HTTP MockTransport 验证请求结构、空输出、截断、格式错误、鉴权错误、超时和非法向量等。真实模型可用性、成本、延迟和提取质量仍需凭证联调。
- **Docker 容器运行**：本机 Docker Desktop Linux 引擎管道不存在；Dockerfile/Compose 已准备并静态校验，但未成功构建或启动容器。
- **生产质量与规模**：没有正式 benchmark、压力测试、集群、TLS 网关或完整用户身份系统。首次向量缓存采用有界渐进建立；默认词法不是完整语义检索。

## 复验命令

```text
uv sync --extra dev --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -q
uv run python scripts/smoke.py
uv build
```

受限 Windows Temp/cache 的处理方式见 [运行维护](operations.md)。验收环境使用项目内 `.tmp` 作为 pytest basetemp，测试不读取或修改真实用户数据库。
