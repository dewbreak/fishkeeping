# 运行与维护

Fishkeeping 首版面向单机可信后端。数据库使用 SQLite，默认监听 `127.0.0.1:8765`。核心操作无需模型凭证；配置 DeepSeek 或 embedding 后才会触发对应网络调用。

## 本地安装

需要 Python 3.11+，推荐 3.12。以下命令均在项目根目录执行。采用 uv 的锁定依赖方式：

```text
uv sync --extra dev --locked
uv run fishkeeping serve
```

首次启动前复制配置。PowerShell：

```powershell
Copy-Item .env.example .env
uv sync --extra dev --locked
uv run fishkeeping serve
```

macOS / Linux：

```bash
cp .env.example .env
uv sync --extra dev --locked
uv run fishkeeping serve
```

无 uv 时，也可以使用 Python 虚拟环境。PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\fishkeeping serve
```

macOS / Linux：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/fishkeeping serve
```

`pip install` 按 `pyproject.toml` 版本范围解析，不保证与 `uv.lock` 完全一致。正式复现优先使用 `uv sync --locked`。进程读取当前工作目录的 `.env`，系统环境变量可以覆盖该文件；相对数据库路径同样相对于工作目录。

## 配置

完整模板在 [`.env.example`](../.env.example)。不使用的可选 key/URL 保持未设置，不要填空字符串作为“关闭”。

| 环境变量 | 默认值 / 用途 |
| --- | --- |
| `FISHKEEPING_DB_PATH` | `data/fishkeeping.sqlite3`；建议服务管理器中使用绝对路径 |
| `FISHKEEPING_HOST`、`FISHKEEPING_PORT` | `127.0.0.1`、`8765` |
| `FISHKEEPING_API_KEYS` | `{}`；JSON 对象，随机 token → namespace，每个 token 至少 16 字符 |
| `FISHKEEPING_DEEPSEEK_API_KEY` | 未设置；启用可选模型能力 |
| `FISHKEEPING_DEEPSEEK_BASE_URL`、`FISHKEEPING_DEEPSEEK_MODEL` | `https://api.deepseek.com`、`deepseek-flash` |
| `FISHKEEPING_EMBEDDING_API_KEY/BASE_URL/MODEL` | 未设置；独立配置 embedding 服务 |
| `FISHKEEPING_REQUEST_TIMEOUT` | 45 秒；范围 1～300 |
| `FISHKEEPING_CANDIDATE_LIMIT` | 5000；范围 100～100000，限制候选扫描数量 |
| `FISHKEEPING_EMBEDDING_MAX_NEW_PER_RECALL` | 64；范围 1～1024，限制单次召回新增向量数量 |

生成服务 token 的一个办法是 `python -c "import secrets; print(secrets.token_urlsafe(32))"`，然后将结果放进 `.env` 的 JSON 映射。不要使用示例占位符作为实际凭证，不要将 `.env` 或数据文件提交到 Git。

如果监听到其他机器可达的地址，应在反向代理提供 TLS、请求大小限制、限速和业务用户授权。namespace key 只是服务边界；该 namespace 内具体用户的授权由业务后端负责。SQLite 数据默认没有静态加密，文件权限和磁盘加密由部署环境管理。

## Docker Compose

先复制 `.env.example` 为 `.env`，再执行：

```text
docker compose config --quiet
docker compose up --build -d
docker compose ps
docker compose logs --tail=100 fishkeeping
```

镜像使用 Python 3.12、非 root UID/GID `10001:10001`；数据写入 `/app/data` 的命名卷。容器内监听 `0.0.0.0`，Compose 只映射宿主 `127.0.0.1:8765`。`.env` 经运行时传入，不复制到镜像。健康检查通过标准库 urllib 调用 `/readyz`，不依赖 curl。配置参考 [Dockerfile 官方说明](https://docs.docker.com/reference/dockerfile/)。

命名卷初次创建会继承镜像中目录的所有者；如果改用 bind mount，需确保宿主目录对 UID 10001 可写。`docker compose down` 停止服务并保留命名卷；增加 `-v` 会删除数据卷，不用于普通更新。当前 Dockerfile 使用 `pip install .`，依赖按版本范围解析；本地/CI 锁定安装以 `uv.lock` 为准。Docker 和真实提供商是否实际运行，以项目验收记录为准，不因提供了配置文件而视为已验证。

## 健康检查与测试

```text
uv run python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8765/healthz').read().decode())"
uv run python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8765/readyz').read().decode())"
uv run --locked ruff check .
uv run --locked pytest -q
uv build
```

`healthz` 检查服务存活，`readyz` 检查数据库；二者不证明 DeepSeek 凭证、额度或模型响应正常。CI 配置覆盖 Ubuntu/Windows 和 Python 3.11/3.12，执行锁定安装、lint、测试和打包；仅配置 CI 不代表远程检查已运行。使用固定 action SHA 和指定 uv 版本，参照 [uv 官方 GitHub Actions 指南](https://docs.astral.sh/uv/guides/integration/github/)。

`uv run python scripts/smoke.py` 会在临时端口启动真实 HTTP 服务，运行 SDK 写入/召回/来源删除检查并停止服务；测试数据只在临时目录中保存。

如果受限 Windows 执行环境无法写入全局 uv cache/Temp，可将 `UV_CACHE_DIR`、`TEMP`、`TMP` 指向项目内已创建的 `.uv-cache` / `.tmp`，设置 `UV_LINK_MODE=copy`。pytest 的临时目录可用 `--basetemp=.tmp/pytest` 指定；这些目录已被 Git 忽略。

## 归档维护

维护是显式操作，没有自动后台定时归档。先请求：

```json
{
  "scope": {"namespace": "personal", "user_id": "alice", "agent_id": "assistant"},
  "dry_run": true,
  "retention_threshold": 0.1
}
```

发送到 `POST /v1/maintenance`，查看候选与统计后，再以 `dry_run=false` 执行。默认 `dry_run=true`。衰减影响排序，归档改变检索可见状态，过期按有效期排除；三者不等价于从磁盘安全擦除。已被新事实替代的数据可通过包含 inactive 的管理读取查看，但不会作为当前事实正常召回。

## 备份、恢复与删除

运行中的 SQLite WAL 数据库可能把新事务保存在 `-wal` 文件中，不能只复制主 `.sqlite3` 文件。优先使用 SQLite 的 Online Backup API；它会生成一致性快照，避免手工拼接主文件和 WAL。依据：[SQLite Backup API](https://www.sqlite.org/backup.html)。

以下脚本可从项目根目录执行，使用当前配置的数据库路径；目标文件按时间命名，避免覆盖已有备份：

```python
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from fishkeeping.config import Settings

source_path = Path(Settings().db_path).resolve()
if not source_path.is_file():
    raise SystemExit(f"Database not found: {source_path}")
Path("backups").mkdir(exist_ok=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
target_path = Path("backups") / f"fishkeeping-{stamp}.sqlite3"
source = sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True)
target = sqlite3.connect(target_path)
try:
    source.backup(target)
    result = target.execute("PRAGMA integrity_check").fetchone()[0]
    if result != "ok":
        raise RuntimeError(result)
finally:
    target.close()
    source.close()
print(target_path)
```

恢复时先停止所有访问该数据库的进程，保留当前数据库及其 `-wal/-shm` 的完整备份，再将快照恢复为主文件；不要让旧 WAL 与恢复的主文件混用。可在一个新的数据目录恢复快照、修改 `FISHKEEPING_DB_PATH` 后启动，以保留原目录用于回退。备份应定期实际恢复并校验作用域统计，不能只检查文件是否存在。

归档属于逻辑状态变更。用户主动忘记会移除应用可用的记录并处理派生关系，但不承诺磁盘取证意义的安全擦除：WAL、空闲页、操作系统快照、备份以及已发送到外部提供商的数据可能另有副本。需要彻底清除时，应同时执行部署环境的数据保留策略、备份轮换和介质级处理；不要把一次 DELETE 请求视为已清除所有副本。

## 容量和升级

首版采用单机 SQLite 与有界候选检索，未声明百万规模或多实例并发写能力。候选限制意味着较早记忆可能不进入召回候选；应用应处理响应警告，并在真实数据上测量延迟和召回质量后决定是否提升上限或迁移存储。

升级前备份，安装后运行测试和健康检查，再验证代表性用户的写入/召回/上下文流程。保留旧程序版本及数据库快照，以便一起回滚；不假设任意旧程序都能读取未来数据库 schema。日志避免记录完整用户记忆、X-API-Key header 或提供商凭证。
