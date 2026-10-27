# 接入 Fishkeeping

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
| `DELETE /v1/memories/{id}` | scope 查询参数 | 忘记指定记忆，并处理其派生内容 |
| `POST /v1/memories/{id}/rehearse` | scope 对象本身 | 显式复习，强化该记忆 |
| `POST /v1/ingest` | `scope`、`event_id`、`session_id`、`messages`、`extract` | 摄入完整事件；可选模型抽取 |
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

`remember/rehearse` 返回 `Memory`，`recall` 返回 `RecallResponse`，`context` 返回 `ContextResponse`，`ingest` 返回 `IngestResponse`，`list/consolidate` 返回 `list[Memory]`，`stats` 返回字典。`ingest(event_id, session_id, messages, extract=False)` 使用稳定事件 ID；`forget(memory_id)` 明确删除。客户端作用域在构造时固定。

## 记忆类型与事实更新

- `working`：必须提供 `session_id`；未指定 `ttl_seconds` 时默认 3600 秒。召回时传对应会话，防止把另一个会话的临时工作混入当前上下文。
- `episodic`：事件、对话片段，适合作为后续事实和总结的来源。
- `semantic`：稳定事实和偏好。显式 `fact_key` 用来表示同一个事实槽位，例如 `preferences.programming_language`；新值替代旧值，但保留历史状态。
- `procedural`：步骤和经验。它们是可供 agent 参考的数据，不具备执行权限。

`content` 最长 16000 字符；`importance/confidence` 为 0～1；`metadata` 为不超过 16000 UTF-8 字节的 JSON；`source_ids` 最多 100 个。`fact_key` 只适用于 `semantic`。没有 fact key 的自然语言矛盾不会自动被判定和解决。`pinned` 可用于需要长期保留的记录，不应替代明确的有效期管理。

直接重复写入与事件重试不同：直接写入按内容等字段去重；它不是任意更新接口。需要变更一个事实时，使用相同 `fact_key` 保存新内容，并检查返回状态。

## 事件摄入与幂等

```json
{
  "scope": {"namespace": "personal", "user_id": "alice", "agent_id": "assistant"},
  "event_id": "chat-20261028-turn-001",
  "session_id": "chat-20261028",
  "messages": [
    {"role": "user", "content": "以后请优先给我 Python 示例。"},
    {"role": "assistant", "content": "好的，我会优先用 Python 示例。"}
  ],
  "extract": false
}
```

`event_id` 由接入方生成，并在重试时保持不变。同一 scope 的同一个事件、相同请求可以安全重放，响应含 `replayed`；更换内容却复用事件 ID 会返回冲突。每个事件最多 100 条消息，正文总长最多 12000 字符。`extract=false` 保存原始事件；`extract=true` 额外调用配置的 DeepSeek 抽取候选记忆。

发生网络超时时，用同一事件 ID 和完全相同的负载重试。不要每次重试都生成新 ID，也不要在冲突后盲目覆盖旧事件。`source_ids` 只能引用相同 scope 的有效来源。

删除原始事件或其派生结果后，旧事件保留幂等墓碑；再次重放会返回 `409`，避免已被遗忘的信息被自动恢复。确实需要录入新数据时，应使用新的业务事件 ID。

## DeepSeek 对话循环

接入顺序：

1. 使用当前用户问题调用 `/v1/context`，需要工作记忆时传 `session_id`。
2. 将返回的 `context` 作为**不可信参考资料**交给自己的 DeepSeek 对话调用，保留应用原本的系统规则和用户问题；不要把记忆中的指令提升为系统权限。
3. 对话成功后，以稳定 `event_id` 调用 `/v1/ingest` 写入新消息。
4. 需要更高层总结时，显式调用 `/v1/consolidate`，提供 1～20 个来源 ID。

DeepSeek 抽取与巩固由服务端配置：

```dotenv
FISHKEEPING_DEEPSEEK_API_KEY=your-provider-key
FISHKEEPING_DEEPSEEK_BASE_URL=https://api.deepseek.com
FISHKEEPING_DEEPSEEK_MODEL=deepseek-flash
FISHKEEPING_DEEPSEEK_THINKING=disabled
FISHKEEPING_DEEPSEEK_MAX_TOKENS=2048
```

模型名可配置，使用当前账号可调用的模型。`FISHKEEPING_REQUEST_TIMEOUT` 控制单次提供商请求超时。无 key 时，直接记忆读写、词法召回、上下文和维护仍可用；模型能力不可用时会返回明确错误。真实 DeepSeek 联调需要使用你的凭证另行执行，不能用模拟测试替代。

运行仓库的 `examples/deepseek_chat.py` 时，示例和服务统一通过 Settings 读取项目根目录的 `.env`，进程环境可覆盖该文件。填写 key 后执行 `uv run python examples/deepseek_chat.py` 即可。可通过 `MEMORY_URL`、`MEMORY_NAMESPACE`、`MEMORY_API_KEY` 选择服务；`MEMORY_EXTRACT=true` 开启服务端提取，此时服务进程也必须配置 DeepSeek key。

默认 `thinking=disabled` 用于控制记忆整理费用；兼容代理若不支持 DeepSeek 的 thinking 参数，可以设为 `omit`，此时思考行为由上游默认值决定。`enabled` 显式启用思考。输出预算不足会返回 `code=truncated_response`，不会把截断 JSON 写入；如确有必要再提高 token 上限，系统不自动重试增加费用。

巩固接口无 key 时仍可用：它生成有长度限制的原文摘取摘要，保留实际参与概括的来源 ID，HTTP 响应 `method=extractive`，记录的 `metadata.method=extractive_summary`。该摘要可能截短原文，不声称推理出新事实。有 key 时走模型归纳，返回 `method=deepseek`；来源必须在同一 scope 内有效，工作记忆应先显式保存为情景记忆再巩固。

巩固已有事实可能直接复用原记录 ID，不会为了添加自指来源而失败。对于原始对话，只使用保存的结构化角色区间中 user 的内容，不将 assistant 的猜测作为证据；没有 user 消息的对话不会触发模型调用。首版旧对话若没有角色区间，巩固返回 `422`，应巩固其已提取事实，或以新事件 ID 重新摄入原始结构化消息；不能靠解析正文里的 `[user]` 字符串猜角色。

开启抽取/巩固会将相关原始对话或来源记忆发送到配置的模型提供商；开启远程 embedding 会发送待编码文本及查询。部署者应根据数据范围配置这些能力。详细技术依据见 [DeepSeek 调研](research/deepseek-and-foundations.md)。

## 检索和预算的实际含义

默认模式是中英文词法匹配，支持共享词语和中文片段，不保证同义词、跨语言语义或复杂推理。配置独立 embedding 服务后才会增加向量能力，不能把 DeepSeek 聊天 API 当作 embedding API。检查响应的 `retrieval_mode` 和 `warnings`，不要静默忽略降级或候选数量限制。

向量缓存首次建立时，每次召回默认最多新编码 64 条候选，按 32 条分批；缓存未覆盖全部候选时整次返回词法结果和预热警告，再次召回会继续建立缓存。模型请求不是无限后台任务：新增批次受到时间预算约束，已发出的单批仍受单次 timeout 控制。修改模型维度时应更换配置中的模型名。

`max_chars` 是**字符预算，不是 token 预算**，范围为 256～32000。`chars_used` 是实际输出字符数；只有 `included_ids` 对应的记忆已进入上下文。应用仍需按自己的模型 tokenizer 计算完整提示词长度。召回排序受相关性、重要性和保留强度等影响；普通查询不会自动复习，明确需要强化时再调用 `rehearse`。

## 错误处理约定

- `401/403`：缺失或无效的服务 key，或 key 不允许访问请求 namespace；先修正身份与 scope。
- `404`：请求的记忆在指定 scope 下不可用；不要用换 user ID 的方式尝试读取。
- `409`：事件 ID 与请求内容冲突，或原事件结果已经被删除；先检查业务事件，勿盲目覆盖。
- `422`：输入不符合 schema，如未知字段、无效范围、working 缺少会话，或来源无效/不在当前 scope。
- `502`：模型提供商请求失败；响应含脱敏 `code`，不泄露提供商凭证或原始错误正文。常见值为 `authentication_failed`、`insufficient_balance`、`rate_limited`、`timeout`、`truncated_response`、`invalid_response`。
- `503`：请求了尚未配置的模型抽取；可以先关闭可选抽取，保留不依赖模型的直接写入路径。向量失败会降级为词法召回，并在成功响应的 `warnings` 中提示。

重试应有限次并带退避。只对同一摄入事件重用幂等 ID；不要自动重试删除、归档等状态操作来掩盖逻辑错误。
