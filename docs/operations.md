# 运行与维护

Fishkeeping v0.2 面向单机可信后端。数据库使用 SQLite，默认监听 `127.0.0.1:8765`。核心操作无需模型凭证；可靠事件由独立 worker 提取，向量索引通过显式命令建立。默认没有定时提取、自动索引或自动归档任务。

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
| `FISHKEEPING_DEEPSEEK_THINKING` | `disabled`；另支持 `enabled` 和兼容代理用的 `omit` |
| `FISHKEEPING_DEEPSEEK_MAX_TOKENS` | 2048；范围 128～16384，模型输出预算 |
| `FISHKEEPING_WORKER_DAILY_CALL_LIMIT` | 20；范围 0～10000，全数据库 worker 共用的 UTC 每日尝试上限，0 暂停领取 |
| `FISHKEEPING_WORKER_MAX_ATTEMPTS` | 3；范围 1～10，新建任务的总尝试上限 |
| `FISHKEEPING_EMBEDDING_BACKEND` | `remote`；无 URL/model 时不启用向量，设 `local` 使用 CPU 模型 |
| `FISHKEEPING_EMBEDDING_API_KEY/BASE_URL/MODEL` | 未设置；远程 embedding 独立配置，URL 与 model 必须同时提供 |
| `FISHKEEPING_LOCAL_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`；当前仅支持此固定模型 |
| `FISHKEEPING_LOCAL_EMBEDDING_CACHE` | `.cache/embeddings`；模型下载缓存目录 |
| `FISHKEEPING_LOCAL_EMBEDDING_THREADS` | 2；范围 1～32，CPU 推理线程数 |
| `FISHKEEPING_LOCAL_EMBEDDING_MIN_SIMILARITY` | 0.50；本地模型语义候选余弦门槛 |
| `FISHKEEPING_EMBEDDING_MIN_SIMILARITY` | 0.35；远程后端门槛，需按提供商和业务数据另行评估 |
| `FISHKEEPING_REQUEST_TIMEOUT` | 45 秒；范围 1～300 |
| `FISHKEEPING_CANDIDATE_LIMIT` | 5000；范围 100～100000，分别限制进入重排的最高分词法/向量候选数量，不截断全范围搜索 |

旧配置 `FISHKEEPING_EMBEDDING_MAX_NEW_PER_RECALL` 已移除；recall 不再自动构建文档向量。运行前移除旧配置并按下节显式建立索引。

生成服务 token 的一个办法是 `python -c "import secrets; print(secrets.token_urlsafe(32))"`，然后将结果放进 `.env` 的 JSON 映射。不要使用示例占位符作为实际凭证，不要将 `.env` 或数据文件提交到 Git。

如果监听到其他机器可达的地址，应在反向代理提供 TLS、请求大小限制、限速和业务用户授权。namespace key 只是服务边界；该 namespace 内具体用户的授权由业务后端负责。SQLite 数据默认没有静态加密，文件权限和磁盘加密由部署环境管理。

## 后台提取与费用上限

API 的 `POST /v1/events` 在同一事务中保存原始情景记忆、幂等凭据和任务，返回 `202`，默认 `extract=true`。无需 DeepSeek key 即可接收并保留 pending 事件。`extract=false` 不生成提取任务；兼容 `/v1/ingest` 仍是同步原子路径，开启提取后模型失败会整体不落库。接入和任务状态约定见 [接入说明](integration.md#可靠事件摄入与后台提取)。

在单独的终端或进程管理器中运行 worker，保证工作目录、数据库路径和模型配置与 API 一致：

```text
uv run fishkeeping worker --once
uv run fishkeeping worker --max-jobs 20 --poll-interval 2
```

`--once` 最多处理一次就绪任务，无任务时立即退出。常规模式无就绪任务时轮询；`--max-jobs` 默认为 20，计数包含失败尝试和无需模型便完成的事件，并不表示 20 个不同业务事件。达到此上限或每日预算耗尽后进程退出，需要再次显式启动或由部署者的进程管理器调度。没有运行 worker 就不会后台提取。轮询不代表提供商重试；模型错误是否再次调用由任务状态决定。

`FISHKEEPING_WORKER_DAILY_CALL_LIMIT` 默认 20，按 UTC 日切换，存于数据库；同一数据库的所有 worker 和 namespace 共用额度，重启不重置。领取任务时在 SQLite 写事务内扣减预算、增加 attempts 并设置租约；崩溃也不回退额度。该数值是保守的模型调用尝试上限，不是成功调用数或金额。worker 应统一配置相同的每日上限，避免不同进程的额度策略不一致。同步 `/v1/ingest`、模型巩固、应用对话和远程 embedding 不受此额度约束。

新任务默认最多尝试 3 次，创建后保留其 `max_attempts`；改变环境变量影响新任务，不重置既有失败任务。429 与 5xx 使用有限指数退避；认证/余额错误、超时、网络错误、输出无效或本地提交失败不会自动重试。worker 崩溃后租约默认 120 秒（`request_timeout × 2 + 30`）到期，下次队列操作将 processing 标记为 `failed/outcome_unknown`，防止自动重复可能已计费的请求。系统保证本地领取和结果提交的一致性，不保证上游调用 exactly-once。

查看 `GET /v1/events/{event_id}` 的 `job.state/last_error_code/attempts`，确认错误原因与再次调用成本后，再向 `/v1/events/{event_id}/retry` POST scope 对象。手动重试仍受单任务总次数和当天预算约束；达到上限返回 `409`。重复 enqueue 不重新排队。来源删除、归档或过期时，未完成任务取消，正在返回的模型结果也不会使来源或派生事实复活。

worker 的标准输出仅含任务 ID、状态、尝试次数、脱敏错误码和每日预算摘要。日志采集不应额外打印请求正文或 `.env`。未配置 key 时 worker 直接退出，API 保存的任务仍在；不存在隐式免费模型替代。

## 本地多语言向量与显式索引

本地后端采用 FastEmbed + ONNX Runtime CPU，模型为 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，输出 384 维向量。下载来源固定为 `Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q`，修订 `faf4aa4225822f3bc6376869cb1164e8e3feedd0`；模型 key 同时包含库版本、修订、归一化和分块策略，避免静默混用不同编码。首次编码按需下载约 252 MB 模型资源，部署时还需预留依赖、缓存和运行内存；本地推理无模型 API 调用费。

安装额外依赖：

```text
uv sync --extra dev --extra local --locked
```

在 `.env` 中选择本地后端：

```dotenv
FISHKEEPING_EMBEDDING_BACKEND=local
FISHKEEPING_LOCAL_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
FISHKEEPING_LOCAL_EMBEDDING_CACHE=.cache/embeddings
FISHKEEPING_LOCAL_EMBEDDING_THREADS=2
```

随后启动服务、显式建立某个 scope 的索引：

```text
uv run --extra local fishkeeping serve
uv run --extra local fishkeeping index --namespace personal --user-id alice --agent-id assistant --batch-size 64 --max-batches 20
```

两条命令在不同终端执行。配置本地向量后，每条 `uv run` 都保留 `--extra local`，包括 worker、示例和测试命令；否则 uv 的同步行为可能移除可选依赖。需要测试工具时同时保留 dev，例如 `uv run --extra dev --extra local --locked pytest -q`。pip 环境对应 `pip install -e '.[dev,local]'`，并使用同一虚拟环境运行后续命令。

模型延迟加载，启动 API 不会立即下载。预部署时可先在可联网环境建立一次索引，再复用完整缓存；下载失败返回脱敏提供商错误，不应通过不断 recall 进行重试。长文本按不超过 126 个内容 token 的区间分块，分块向量按 token 长度加权平均并再次归一化，避免仅保留开头；这仍是单个文档向量，不保证长篇多主题的全部细节都能精确命中。

索引命令 `--batch-size` 默认 64，每个请求最多处理 1024 条；内部每批最多 32 条且累计最多 32000 字符，批次间检查 `FISHKEEPING_REQUEST_TIMEOUT` 时间预算。已启动的本地加载/编码不会被时间预算硬中断；远程单批仍受 HTTP timeout 约束。`--max-batches` 默认 20，达到上限、覆盖完成或本轮没有写入后退出，输出每轮 `processed/total/indexed/complete/model_key`。HTTP 对应 `POST /v1/index`，输入 `limit` 范围 1～1024、默认 64；可分多次续建。

需要清理当前 scope、当前模型的旧向量并重新建立时：

```text
uv run --extra local fishkeeping index --namespace personal --user-id alice --agent-id assistant --batch-size 64 --max-batches 20 --rebuild
```

CLI 仅第一批发送 `rebuild=true`，后续批次继续补齐；HTTP/SDK 调用方也必须这样处理，不能每批都清空缓存。重建不删除记忆，不清理其他 scope 或模型 key 的缓存。清理后编码失败会留下尚未完成的索引，按 `rebuild=false` 续建即可。

recall 只编码查询，不编码缺失文档；没有任何缓存时使用词法结果并提示未建索引。部分覆盖返回 `hybrid_partial` 与覆盖率警告，覆盖完整返回 `hybrid`。新增记忆后应安排显式增量索引，避免误把局部覆盖当作完整语义搜索。查询失败会降级为词法并给出警告。

若选择远程服务，保留 `FISHKEEPING_EMBEDDING_BACKEND=remote`，同时设置独立的 `FISHKEEPING_EMBEDDING_BASE_URL/MODEL`，按需配置其 key；默认模板未设置 URL/model，所以仍保持纯词法兼容。远程文档编码和查询可能计费，应自行限制索引批次、调用频率和服务额度。DeepSeek 聊天 key 不会自动作为 embedding key 使用。

以上命令说明运行方式；本地模型、Docker 和真实 DeepSeek 的实际验证范围以 [验收记录](verification.md) 为准。

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

默认镜像不安装 `local` extra，不包含 FastEmbed 本地向量运行时；需要本地模型时使用上述宿主环境，或自行构建安装 `.[local]` 并配置缓存目录的镜像。Compose 默认只启动 API，没有自动启动 worker；需要部署者另行配置共享同一数据卷的提取进程。不能将宿主上的模型验证结果视为容器兼容性验证。

## 健康检查与测试

```text
uv run python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8765/healthz').read().decode())"
uv run python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8765/readyz').read().decode())"
uv run --extra dev --locked ruff check .
uv run --extra dev --locked pytest -q
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

词法检索使用 SQLite FTS5，对完整 scope 的合格记录搜索后按相关度取最高分候选；早期记忆不再因为“只取最新 N 条”被预先排除。向量检索精确扫描当前模型下的全部合格缓存，再保留有限最高分候选；它是线性扫描，内存中保留的候选有界，但总扫描耗时随数据规模增长。达到候选上限会返回警告。当前仍是单机 SQLite 方案，多 worker 的领取由事务协调，不等于声明百万规模或多主写入能力。

每次写入最多提交 100 个来源，重复合并后的完整来源可以超过 100。`Memory.source_ids` 是最多 100 条的预览，`source_count` 是完整数量；完整来源通过 `/v1/memories/{id}/sources` 分页读取。关系表负责完整删除传播和失效检查，预览不会截断或覆盖历史来源。

核心数据库 schema 已升级为 `PRAGMA user_version=2`。首次以 v0.2 打开旧数据库时，单个事务回填 FTS 索引与来源关系，成功后才记录新版本；失败回滚并拒绝启动。运行时 SQLite 必须支持 FTS5，缺失时启动报错。队列表另有版本检查，不接受未知未来版本。

升级前停止 worker 与旧 API，创建一致性备份，再安装新版本并启动。迁移可能需要扫描所有历史文档，应预留时间与磁盘空间，避免在迁移期间让旧程序继续写库。旧 v0.1 程序拒绝 schema 2，不能直接降级连接迁移后的数据库；回滚时将旧程序和升级前数据库快照一起恢复。之后运行测试、健康检查及代表性用户的事件摄入、来源分页、召回流程。日志避免记录完整用户记忆、X-API-Key header 或提供商凭证。
