# MCP 接入：按需开放记忆、识图和视频

Fishkeeping 提供独立 MCP 服务端，供支持 MCP 的客户端调用。不需要先启动 REST API；它直接复用现有记忆存储、检索、OCR 和视频采样逻辑。默认只开放记忆工具，`--modules` 决定客户端能看到哪些工具。切换模块后重启该 MCP 服务即可。

## 安装与启动

只使用记忆：

```text
uv sync --extra dev --extra mcp --locked
uv run --extra mcp --locked fishkeeping mcp --user-id alice
```

开启图片、视频时添加 `vision` 依赖，三组可任意组合：

```text
uv sync --extra dev --extra mcp --extra vision --locked
uv run --extra mcp --extra vision --locked fishkeeping mcp --user-id alice --modules memory image video
```

| 需要的能力 | 参数 | 客户端工具 |
| --- | --- | --- |
| 只要记忆 | `--modules memory`（默认） | `memory_remember`、`memory_recall`、`memory_context`、`memory_list`、`memory_get`、`memory_forget`、`memory_sources` |
| 只要识图 | `--modules image` | `image_ocr`、`image_view` |
| 只要视频 | `--modules video` | `video_sample` |
| 图片与视频 | `--modules image video` | 两组媒体工具，不开放记忆工具 |
| 全部 | `--modules memory image video` | 上述三组工具 |

未启用的工具不注册，不会先出现在工具列表再在调用时拒绝。模块设置只影响本 MCP 进程，不改变现有 REST API、SDK 或其他 MCP 进程。

默认传输为 `stdio`，由 MCP 客户端启动进程并连接标准输入/输出。直接在终端启动会等待协议消息，不能像普通交互命令一样输入聊天文字。标准输出只承载 MCP 协议；诊断信息写到标准错误。

## 客户端配置

下面是采用 `mcpServers` 配置格式的客户端示例。项目位置请换成自己的路径；`uv --directory` 也让服务从项目目录读取 `.env`、数据库相对路径和默认媒体根目录。

```json
{
  "mcpServers": {
    "fishkeeping": {
      "command": "uv",
      "args": [
        "--directory", "D:/Task/fishkeeping",
        "run", "--extra", "mcp", "--extra", "vision", "--locked",
        "fishkeeping", "mcp", "--user-id", "alice",
        "--modules", "memory", "image", "video",
        "--media-root", "D:/Task/fishkeeping"
      ]
    }
  }
}
```

如果希望在客户端界面分别开关三个服务，可以复制这个配置为 `fishkeeping-memory`、`fishkeeping-image`、`fishkeeping-video` 三个条目，每个条目的 `--modules` 后只保留同名模块。记忆条目可移除 `--extra vision`；媒体条目保留。三个条目是三个独立进程，客户端可分别启停，工具名不会重叠。仅需一个组合服务时保留上面的单条目即可。

启动参数 `--namespace`、`--user-id` 和 `--agent-id` 固定该进程的 scope；其中 user-id 必填，另两项默认都是 `default`。模型调用工具时无需重复传 scope，也不能通过工具参数切换用户。

| 参数 | 默认值/用途 |
| --- | --- |
| `--user-id` | 必填；记忆所属用户 |
| `--namespace` | `default` |
| `--agent-id` | `default` |
| `--media-root` | 当前工作目录；允许读取图片/视频的根目录 |
| `--modules` | `memory` |
| `--transport` | `stdio` |

记忆仍使用 `.env` 中的 `FISHKEEPING_DB_PATH`，与 REST API 可指向同一个 SQLite 文件。MCP 的记忆工具不运行提取 worker、巩固或 DeepSeek 问答；`memory_remember` 直接保存调用方提供的内容。检索沿用已有 embedding 配置；如果明确配置了远程 embedding，检索仍可能产生对应提供商费用。本地向量的安装和索引方式见 [运行维护](operations.md)。

如果 `.env` 设置了 `FISHKEEPING_EMBEDDING_BACKEND=local`，安装、启动及客户端 `args` 都应保留 `--extra local`，例如 `uv run --extra mcp --extra local --locked fishkeeping mcp --user-id alice`。同时使用媒体时再加 `--extra vision`。索引仍通过原有 `fishkeeping index` 或 `index-worker` 显式构建。

## 工具调用与媒体成本

图片和视频工具接受本地 `path`，不要求客户端把整个文件编码进工具参数。相对路径相对于 `--media-root`，绝对路径也必须位于该目录内。文件在 MCP 服务所在机器上读取；HTTP 客户端本机的路径不能自动传到远端服务器。

- `image_ocr`：传入 `path`、`question`，复杂表格可加 `layout="table"`；返回 OCR 文本、坐标、来源和复核标记。它不会把图片内容自动保存为事实。
- `image_view`：返回原生 MCP 图片块供客户端模型观察，适合场景、空间关系或 OCR 不能回答的问题。默认 `detail="low"` 将最长边限制为 512；`detail="original"` 保留解码图片的当前尺寸。两者都输出规范化 JPEG 预览，来源与预览分别记录哈希。
- `video_sample`：传入 `path`、`question`，按需加 `start_seconds`、`end_seconds`、`max_frames`；返回带时间戳的选中帧与来源信息。默认最多六帧、`detail="low"` 的 512 边长预览；`detail="original"` 使用解码选中帧的当前尺寸，并非原始视频像素的无损传输。范围和稀疏采样限制与现有视频管线一致。

例如请求视频中的特定时段：

```json
{
  "path": "clips/inspection.mp4",
  "question": "只看13至17秒，角色按什么顺序出现？",
  "start_seconds": 13,
  "end_seconds": 17,
  "max_frames": 6
}
```

返回的像素使用 MCP `ImageContent`，不会同时在普通 JSON 中重复一份帧 Base64。MCP 服务本身不另外调用 DeepSeek；图片识别和最终视频回答由客户端模型基于这些证据完成，因此客户端需要能处理 MCP 图片块。客户端仍可能为模型输入计费，少帧、明确窗口和小预览只是控制输入量，不能保证任意模型相同的 token 降幅。

`image_ocr` 返回现有紧凑审计结构：`regions` 每项为 `[id, [x1,y1,x2,y2], text, confidence]`，不再嵌套重复的 `packed` 文本。`memory_list` 的结构化结果由官方 SDK 包装为 `{"result": [...]}`，其余记忆对象按原有字段返回。文本内容和 `structured_content` 是 MCP 的两种访问方式，客户端不应把它们重复加入模型提示。

OCR 的 `requires_review`、视频时间间隙和未采样内容仍应交给客户端判断。抽帧没有音频，不证明帧间动作或事件未发生。需要记住识别结果时，确认内容后再显式调用记忆工具。现有媒体限制、已测效果与失败案例见 [图片指南](vision.md)、[视频指南](video.md) 和 [真实素材评测](media-evaluation.md)。

## Streamable HTTP

需要常驻的本地 HTTP MCP 端点时：

```text
uv run --extra mcp --extra vision --locked fishkeeping mcp --user-id alice --modules memory image video --transport streamable-http --host 127.0.0.1 --port 8766
```

客户端连接 `http://127.0.0.1:8766/mcp`。该端点使用无状态 HTTP 传输；记忆持久性仍由 SQLite 提供，与 HTTP 会话是否保存无关。不同客户端的 HTTP 配置字段可能不同，填写同一端点即可。

此服务按启动时的固定 scope 面向受信任本地客户端，不复用 REST 的 `X-API-Key` 认证。若需跨机器部署，应由现有可信网关完成身份认证和访问控制；当前示例保持回环绑定。官方 SDK 接口说明见 [MCP Python SDK](https://py.sdk.modelcontextprotocol.io/)。

## 本地复验

```text
uv run --extra dev --extra mcp --locked pytest tests/test_mcp.py tests/test_mcp_cli.py -q
uv run --extra dev --extra mcp --locked python scripts/smoke_mcp.py
```

第一条验证选择模块和工具的实际协议调用；第二条启动真正的 stdio 子进程和回环 HTTP 服务并完成记忆/图片流程。两者不调用付费模型，不需要 OCR 权重或下载视频。需要保留已安装的 local/vision 功能时，在命令中同时添加对应 extra。最新结果见 [验收记录](verification.md)。
