# 接入 Fishkeeping

v0.3 新增 `POST /v1/images/analyze`、`POST /v1/images/events` 及 SDK `analyze_image()` / `enqueue_image()`。输入字节由客户端转为 Base64 JSON，服务只作本地 OCR，不会隐式调用 DeepSeek；类型、幂等、低置信度回退和示例见 [图片接入指南](vision.md)。图片事件作为工具证据保存，确认后的事实通过 `source_ids=[episode.id]` 关联。持续补索引进程见 [index-worker](research/incremental-index.md)。

Fishkeeping 提供独立 HTTP API，适合在现有 agent 的“调用模型前召回、对话完成后写入”两个位置接入。服务启动后，交互文档位于 `http://127.0.0.1:8765/docs`，机器可读规范位于 `/openapi.json`。部署方式见 [运行维护](operations.md)。

## 作用域和认证

每次数据操作必须提供 `scope`：

```json
{"namespace": "personal", "user_id": "alice", "agent_id": "assistant"}
```

三个字段共同决定记忆边界。`user_id` 必填，`namespace` 和 `agent_id` 默认 `default`；建议接入方始终明确传入三者。标识符最长 128 字符，只支持字母数字、下划线、Unicode 单词字符及 `.:-`。读取和删除的 scope 使用查询参数，写入等接口使用 JSON 字段。

配置了 `FISHKEEPING_API_KEYS` 时，发送 `X-API-Key: <token>`。一个 token 绑定一个 namespace。**这不是最终用户登录系统**：同一 namespace 内的 `user_id` 和 `agent_id` 由可信后端选择，业务网关必须根据自己的登录态确定它们，不能让浏览器随意指定其他用户 ID。

没有配置 key 映射时服务不鉴权，只适合受信任的本机。客户端携带的 Fishkeeping token 与服务器调用 DeepSeek 的提供商 key 是不同凭证。

## 最小 HTTP 示例

以下 Python 示例不依赖 DeepSeek key。先启动服务，再在另一个终端执行；`httpx` 已是项目依赖。

```python
import httpx

scope = {"namespace": "personal", "user_id": "alice", "agent_id": "assistant"}
with httpx.Client(base_url="http://127.0.0.1:8765", timeout=60) as client:
    # 如已启用认证，给 Client 增加 headers={"X-API-Key": "..."}。
    response = client.post(
        "/v1/memories",
        json={
            "scope": scope,
            "kind": "semantic",
            "content": "Alice 喜欢用 Python 写数据处理脚本。",
            "fact_key": "preferences.programming_language",
            "importance": 0.8,
        },
    )
    response.raise_for_status()
    response = client.post(
        "/v1/context",
        json={
            "scope": scope,
            "query": "Alice 喜欢用什么语言写脚本？",
            "limit": 5,
            "max_chars": 3000,
        },
    )
    response.raise_for_status()
    recalled = response.json()
    print(recalled["context"])
    print(recalled["included_ids"], recalled["chars_used"])
```

## 接口一览

| 方法与路径 | 输入 | 作用 |
| --- | --- | --- |
| `POST /v1/memories` | `MemoryInput` | 直接保存一条记忆 |
| `GET /v1/memories` | scope 查询参数、`include_inactive`、`limit`、`offset` | 分页列出记忆 |
| `GET /v1/memories/{id}` | scope 查询参数、可选 `include_inactive` | 读取单条记忆 |
| `GET /v1/memories/{id}/sources` | scope 查询参数、`limit`、`offset` | 分页读取完整来源 ID |
| `DELETE /v1/memories/{id}` | scope 查询参数 | 忘记指定记忆，并处理其派生内容 |
| `POST /v1/memories/{id}/rehearse` | scope 对象本身 | 显式复习，强化该记忆 |
| `POST /v1/events` | `scope`、`event_id`、`session_id`、`messages`、可选 `extract` | 可靠摄入：先提交原始事件与提取任务，返回 `202` |
| `GET /v1/events/{event_id}` | scope 查询参数 | 查询原始事件与最新任务状态 |
| `POST /v1/events/{event_id}/retry` | scope 对象本身 | 有限次手动重试失败任务 |
| `POST /v1/ingest` | `scope`、`event_id`、`session_id`、`messages`、`extract` | 兼容同步原子摄入；模型失败时整体不落库 |
| `POST /v1/index` | `scope`、可选 `limit/rebuild` | 显式建立当前向量模型的文档索引 |
| `POST /v1/recall` | `scope`、`query`、可选 `session_id/kinds/limit` | 返回命中、分数、模式与警告 |
| `POST /v1/context` | recall 参数，加 `max_chars` | 组装有预算的上下文 |
| `POST /v1/consolidate` | `scope`、`source_ids` | 对指定来源做抽取式摘要或模型巩固 |
| `POST /v1/maintenance` | `scope`、`dry_run`、`retention_threshold` | 预览或执行归档维护 |
| `GET /v1/stats` | scope 查询参数 | 查看该作用域统计 |
| `GET /healthz`、`GET /readyz` | 无 | 进程健康和数据库就绪检查 |

接口具体响应以当前 `/openapi.json` 为准，仓库保留 [OpenAPI 快照](openapi.json)。公开 JSON 请求体禁止未知字段，以便尽早发现拼写或版本不兼容。

## Python 客户端

安装本项目后，可直接使用同步 HTTP 客户端；它与服务器通信，不绕过服务认证。

```python
from fishkeeping.client import MemoryClient
from fishkeeping.models import Scope

scope = Scope(namespace="personal", user_id="alice", agent_id="assistant")
with MemoryClient("http://127.0.0.1:8765", scope=scope) as memory:
    # 如已启用认证，构造客户端时增加 api_key="..."。
    saved = memory.remember("Alice 喜欢 Python 示例。", kind="semantic")
    result = memory.recall("Alice 喜欢什么示例？")
    for hit in result.hits:
        print(hit.memory.content, hit.relevance, hit.retention)
    context = memory.context("Python 示例", max_chars=3000)
    print(context.context)
    memory.rehearse(saved.id)
```

`remember/rehearse` 返回 `Memory`，`recall` 返回 `RecallResponse`，`context` 返回 `ContextResponse`，`enqueue/event` 返回 `EventResponse`，`retry_event` 返回 `JobInfo`，`sources` 返回 `SourcePage`，`index` 返回 `IndexResponse`。兼容方法 `ingest` 返回 `IngestResponse`；`list/consolidate` 返回 `list[Memory]`，`stats` 返回字典。`forget(memory_id)` 明确删除。客户端作用域在构造时固定。

## 记忆类型与事实更新

- `working`：必须提供 `session_id`；未指定 `ttl_seconds` 时默认 3600 秒。召回时传对应会话，防止把另一个会话的临时工作混入当前上下文。
- `episodic`：事件、对话片段，适合作为后续事实和总结的来源。
- `semantic`：稳定事实和偏好。显式 `fact_key` 用来表示同一个事实槽位，例如 `preferences.programming_language`；新值替代旧值，但保留历史状态。
- `procedural`：步骤和经验。它们是可供 agent 参考的数据，不具备执行权限。

`content` 最长 16000 字符；`importance/confidence` 为 0～1；`metadata` 为不超过 16000 UTF-8 字节的 JSON；每次写入请求的 `source_ids` 最多 100 个。同一事实多次合并后可以保留超过 100 个来源，关系表保存完整集合。`Memory.source_ids` 仅预览最多 100 个 ID，`source_count` 表示完整数量；需要完整证据链时调用来源分页接口。`fact_key` 只适用于 `semantic`。没有 fact key 的自然语言矛盾不会自动被判定和解决。`pinned` 可用于需要长期保留的记录，不应替代明确的有效期管理。

直接重复写入与事件重试不同：直接写入按内容等字段去重；它不是任意更新接口。需要变更一个事实时，使用相同 `fact_key` 保存新内容，并检查返回状态。

## 可靠事件摄入与后台提取

新接入推荐调用 `POST /v1/events`。以下请求先把原始情景记忆、幂等凭据和提取任务提交到同一个 SQLite 事务，再返回 `202`；HTTP 请求不会调用 DeepSeek。`extract` 默认 `true`，即使还未配置模型 key，也可保存事件并等待 worker。

```json
{
  "scope": {"namespace": "personal", "user_id": "alice", "agent_id": "assistant"},
  "event_id": "chat-20261028-turn-001",
  "session_id": "chat-20261028",
  "messages": [
    {"role": "user", "content": "以后请优先给我 Python 示例。"},
    {"role": "assistant", "content": "好的，我会优先用 Python 示例。"}
  ],
  "extract": true
}
```

响应包含 `event_id`、`scope`、`episode`、`job` 和 `replayed`。`episode` 是已持久化的原始事件，`job.state` 初始为 `pending`；`extract=false` 只保存事件，`job=null`。每个事件最多 100 条消息，正文总长最多 12000 字符。

在另一个进程启动 worker，使用与 API 相同的数据库配置：

```text
uv run fishkeeping worker --once
uv run fishkeeping worker --max-jobs 20 --poll-interval 2
```

worker 未运行时不会自动提取。第二条命令最多处理 20 次任务尝试，达到上限或每日预算耗尽后退出；暂无就绪任务时会等待轮询。worker 必须配置 `FISHKEEPING_DEEPSEEK_API_KEY`。使用本地向量额外依赖的环境，将这两条命令改为 `uv run --extra local fishkeeping worker ...`，以保留可选依赖。

用 `GET /v1/events/{event_id}` 携带 scope 查询当前状态。任务状态为 `pending/processing/succeeded/failed/cancelled`，响应包含 `attempts/max_attempts`、下次尝试时间、脱敏错误码及成功的 `result_ids`，不暴露内部租约或提供商错误正文。失败不会丢失原始事件；来源删除、归档或失效后，worker 不会重新生成其事实。

默认每个任务最多 3 次提取尝试，全数据库所有 worker 共用每日 20 次预算，按 UTC 日期切换。预算在领取任务时持久化扣减，即使随后进程崩溃也不退回，所以是保守的调用尝试上限，不是账单核算。该预算只约束后台提取，不涵盖 `/v1/ingest`、巩固、应用自身对话或远程 embedding 的费用。没有用户消息的事件可以直接完成，既不调用模型也不扣预算。

只有 `rate_limited`（429）和 `upstream_unavailable`（5xx）会在尝试上限内自动退避；默认最多退避 2 秒、4 秒。超时、网络错误、格式错误、认证/余额错误和本地提交失败转为 `failed`。processing 租约在 `2 × request_timeout + 30` 秒后失效，转为 `failed/outcome_unknown`，因为上游可能已完成并计费。确认可以再次承担调用费用后，向 `POST /v1/events/{event_id}/retry` 发送 scope 对象；只允许未达到总尝试上限的失败任务，已在 pending 的重复重试保持幂等。手动重试不重置总次数或每日预算。

```python
with MemoryClient("http://127.0.0.1:8765", scope=scope) as memory:
    accepted = memory.enqueue(
        "chat-20261028-turn-001",
        "chat-20261028",
        [{"role": "user", "content": "以后请优先给我 Python 示例。"}],
    )
    print(accepted.episode.id, accepted.job.state)
    latest = memory.event(accepted.event_id)
    print(latest.job.state, latest.job.result_ids)
    # 业务明确批准再次尝试后：memory.retry_event(accepted.event_id)
```

`event_id` 由接入方生成，并在 HTTP 重试时保持不变。同一 scope 的同一个事件、相同请求可以安全重放，响应含 `replayed`；重复摄入只返回当前任务，不重新排队。更换内容却复用事件 ID 返回 `409`。

发生网络超时时，用同一事件 ID 和完全相同的负载重试。不要每次重试都生成新 ID，也不要在冲突后盲目覆盖旧事件。`source_ids` 只能引用相同 scope 的有效来源。

删除原始事件或其派生结果后，旧事件保留幂等墓碑；再次重放会返回 `409`，避免已被遗忘的信息被自动恢复。确实需要录入新数据时，应使用新的业务事件 ID。

兼容接口 `POST /v1/ingest` 保留同步原子语义，`extract` 默认 `false`；开启时在该 HTTP 请求中调用模型，成功后同时提交原始事件与事实，模型失败则整个事件不落库。未配置 key 时该路径的 `extract=true` 返回 `503`。它与可靠摄入使用不同的负载指纹，不能在两个接口间重用同一 scope 下的事件 ID。两条路径都不承诺外部模型调用 exactly-once；例如模型已响应、进程尚未提交就崩溃，后续重试仍可能再次付费。

## 读取完整来源

`GET /v1/memories/{id}/sources` 返回 `memory_id/items/total/limit/offset/has_more`，其中 `items` 是来源 ID。`limit` 默认为 100，范围 1～1000，`offset` 非负；所有查询均要求目标记忆属于当前 scope。

```python
source_ids = []
offset = 0
while True:
    page = memory.sources(memory_id, limit=100, offset=offset)
    source_ids.extend(page.items)
    if not page.has_more:
        break
    offset += len(page.items)
```

上例在已打开的 `MemoryClient` 中执行。分页按稳定 ID 顺序返回当前关系，不是冻结快照；并发写入期间需要审计一致性时，应在业务层安排静默窗口。删除传播仍沿完整关系表处理，不受 100 条预览限制。

## DeepSeek 对话循环

接入顺序：

1. 使用当前用户问题调用 `/v1/context`，需要工作记忆时传 `session_id`。
2. 将返回的 `context` 作为**不可信参考资料**交给自己的 DeepSeek 对话调用，保留应用原本的系统规则和用户问题；不要把记忆中的指令提升为系统权限。
3. 对话成功后，以稳定 `event_id` 调用 `/v1/events` 写入新消息，由独立 worker 后台提取。
4. 需要更高层总结时，显式调用 `/v1/consolidate`，提供 1～20 个来源 ID。

DeepSeek 抽取与巩固由服务端配置：

```dotenv
FISHKEEPING_DEEPSEEK_API_KEY=your-provider-key
FISHKEEPING_DEEPSEEK_BASE_URL=https://api.deepseek.com
FISHKEEPING_DEEPSEEK_MODEL=deepseek-flash
FISHKEEPING_DEEPSEEK_THINKING=disabled
FISHKEEPING_DEEPSEEK_MAX_TOKENS=2048
```

模型名可配置，使用当前账号可调用的模型。`FISHKEEPING_REQUEST_TIMEOUT` 控制单次提供商请求超时。无 key 时，可靠事件保存、直接记忆读写、词法召回、上下文和维护仍可用；worker 暂不提取。真实提供商的验证范围见 [验收记录](verification.md)，模拟测试不能替代真实联调。

运行仓库的 `examples/deepseek_chat.py` 时，示例和服务统一通过 Settings 读取项目根目录的 `.env`，进程环境可覆盖该文件。填写 key 后执行 `uv run python examples/deepseek_chat.py` 即可；本地向量环境使用 `uv run --extra local python examples/deepseek_chat.py`。可通过 `MEMORY_URL`、`MEMORY_NAMESPACE`、`MEMORY_API_KEY` 选择服务。示例通过 `enqueue` 保存事件，但为控制费用默认 `MEMORY_EXTRACT=false`；设为 `true` 才创建后台提取任务，此时需要单独启动已配置 DeepSeek key 的 worker。对话模型调用本身与后台提取是两次独立用途的调用。

默认 `thinking=disabled` 用于控制记忆整理费用；兼容代理若不支持 DeepSeek 的 thinking 参数，可以设为 `omit`，此时思考行为由上游默认值决定。`enabled` 显式启用思考。输出预算不足会返回 `code=truncated_response`，不会把截断 JSON 写入；如确有必要再提高 token 上限，系统不自动重试增加费用。

巩固接口无 key 时仍可用：它生成有长度限制的原文摘取摘要，保留实际参与概括的来源 ID，HTTP 响应 `method=extractive`，记录的 `metadata.method=extractive_summary`。该摘要可能截短原文，不声称推理出新事实。有 key 时走模型归纳，返回 `method=deepseek`；来源必须在同一 scope 内有效，工作记忆应先显式保存为情景记忆再巩固。

巩固已有事实可能直接复用原记录 ID，不会为了添加自指来源而失败。对于原始对话，只使用保存的结构化角色区间中 user 的内容，不将 assistant 的猜测作为证据；没有 user 消息的对话不会触发模型调用。首版旧对话若没有角色区间，巩固返回 `422`，应巩固其已提取事实，或以新事件 ID 重新摄入原始结构化消息；不能靠解析正文里的 `[user]` 字符串猜角色。

开启抽取/巩固会将相关原始对话或来源记忆发送到配置的模型提供商；开启远程 embedding 会发送待编码文本及查询。部署者应根据数据范围配置这些能力。详细技术依据见 [DeepSeek 调研](research/deepseek-and-foundations.md)。

## 检索和预算的实际含义

默认模式是中英文词法匹配，支持共享词语和中文片段，不保证同义词、跨语言语义或复杂推理。FTS5 在整个符合 scope、有效期、类型和会话条件的数据集中查找，按相关度选取候选；不会先截取最新若干条而漏掉较早的相关事件。`FISHKEEPING_CANDIDATE_LIMIT` 默认 5000，分别限制进入重排的词法候选和向量候选数量；超出时返回警告。

向量候选对当前 scope 和模型已缓存的全部合格向量做精确相似度扫描，仅保留最高分候选再与词法结果合并。这避免只搜索最近记录，但扫描时间随向量数量增长，并非 ANN 索引或百万规模性能承诺。检查响应的 `retrieval_mode` 与 `warnings`：`lexical` 表示词法模式，`hybrid` 表示当前过滤范围的向量覆盖完整，`hybrid_partial` 表示使用已建成的部分向量并提示覆盖率。无可用缓存或向量查询失败时返回词法结果及相应警告。

v0.2 不再在 recall 中编码文档或预热缓存，旧配置 `FISHKEEPING_EMBEDDING_MAX_NEW_PER_RECALL` 已移除。请显式请求：

```json
{
  "scope": {"namespace": "personal", "user_id": "alice", "agent_id": "assistant"},
  "limit": 64,
  "rebuild": false
}
```

发送到 `POST /v1/index`；SDK 对应 `memory.index(limit=64, rebuild=False)`。`limit` 范围 1～1024，默认 64，每个编码批次最多 32 条且累计最多 32000 字符，批次之间检查 `request_timeout` 时间预算；已开始的本地模型加载或编码不会被该预算强制中断，远程请求仍受提供商 timeout 约束。响应返回 `processed/total/indexed/complete/model_key`；调用方设置总批次上限并在后续请求继续补齐，避免无界任务。

`rebuild=true` 先清理当前 scope 与当前 model key 的向量缓存，再建立这一批；它不删除原始记忆或其他作用域/模型的缓存。多次补齐时只在第一批设置 `true`，后续保持 `false`。更换远程模型的维度或编码语义时应使用新模型名，或明确重建当前缓存。

本地向量选用 FastEmbed/ONNX CPU 的 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，384 维，固定模型修订；安装 `local` extra 后设 `FISHKEEPING_EMBEDDING_BACKEND=local`。首次使用需下载约 252 MB 模型资源，后续缓存可离线复用；本地编码没有模型 API 调用费，仍消耗 CPU、内存和磁盘。模型原生窗口为 128 tokens，适配器按最多 126 个正文 token 分块，按块长度加权平均并归一化，避免只编码开头。具体安装、缓存和运行命令见 [运行维护](operations.md#本地多语言向量与显式索引)。远程 OpenAI-compatible embedding 仍可独立配置，DeepSeek 聊天 API 不充当 embedding API。

`max_chars` 是**字符预算，不是 token 预算**，范围为 256～32000。`chars_used` 是实际输出字符数；只有 `included_ids` 对应的记忆已进入上下文。上下文中的来源同样提供 `source_count` 与有限 `source_ids` 预览，需要完整证据时调用来源分页。应用仍需按自己的模型 tokenizer 计算完整提示词长度。召回排序受相关性、重要性和保留强度等影响；普通查询不会自动复习，明确需要强化时再调用 `rehearse`。

## 错误处理约定

- `401/403`：缺失或无效的服务 key，或 key 不允许访问请求 namespace；先修正身份与 scope。
- `404`：请求的记忆、来源目标或事件在指定 scope 下不可用；不要用换 user ID 的方式尝试读取。
- `409`：事件 ID 与请求内容/摄入模式冲突、原事件结果已经被删除，或任务不能重试（包括不存在或尝试次数耗尽）；先检查业务事件，勿盲目覆盖。
- `422`：输入不符合 schema，如未知字段、无效范围、working 缺少会话，或来源无效/不在当前 scope。
- `429`：同一视频处理器已有解码任务；调用方应控制并发，不能启动无界重试。
- `502`：模型提供商请求失败；响应含脱敏 `code`，不泄露提供商凭证或原始错误正文。常见值为 `authentication_failed`、`insufficient_balance`、`rate_limited`、`timeout`、`truncated_response`、`invalid_response`。
- `503`：同步抽取或显式索引请求缺少相应提供商配置。可靠 `/v1/events` 仍可保存 pending 任务。向量召回失败会降级为词法模式，并在成功响应的 `warnings` 中提示。

后台提取失败通过 `job.state/last_error_code` 表达，不会把已接受的事件改成 HTTP `502`。先检查任务状态，再决定是否手动重试。

重试应有限次并带退避。只对同一摄入事件重用幂等 ID；不要自动重试删除、归档等状态操作来掩盖逻辑错误。

## 图片与视频证据

图片 `/v1/images/analyze` / `/v1/images/events` 和 SDK `analyze_image` / `enqueue_image` 接受 `layout="table"`，详细用法见 [图片指南](vision.md)。默认 `layout="text"` 保留 v0.3 事件身份；旧事件重放不重新 OCR，改成 table 必须换 event ID。

视频 `/v1/videos/analyze` / `/v1/videos/events` 与 SDK `analyze_video` / `enqueue_video` 接受视频字节、问题、`start_seconds/end_seconds`、`max_frames` 和 `strategy`。analyze 返回带临时 JPEG 的 `VideoEvidence`，events 返回普通 `EventResponse`，episode 内只有来源元数据。事件不生成付费提取任务；模型问答须另外显式调用 `DeepSeekImageChat.answer_video`，见 [视频指南](video.md)。
