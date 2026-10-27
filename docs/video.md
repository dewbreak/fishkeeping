# 低 token 视频问答与来源保存

本阶段把视频转为有时间戳的少量图像，再显式交给 DeepSeek Flash。它是稀疏多帧视觉问答，不是原生视频流理解；当前流程不读取音轨，不提供语音转写或完整连续动作证据。有关官方接口、GitHub 参考方案和素材许可见 [真实媒体调研](research/video-and-real-media.md)，解码与选帧算法详见 [视频管线](research/video-pipeline.md)。

## 本地运行

```text
uv sync --extra dev --extra local --extra vision --locked
uv run --extra local --extra vision python examples/video_question.py clip.mp4 "画面中发生了哪些可见变化？"
```

示例默认在本地解码并打印来源 JSON，不调用模型、不写入记忆数据库，也不把帧 Base64 打印到终端。若问题明确指向某个时间段，可显式指定窗口：

```text
uv run --extra local --extra vision python examples/video_question.py clip.mp4 "只看13至17秒，角色按什么顺序出现？" --start 13 --end 17 --max-frames 6
```

问题中的数字不会被自动解析为窗口；调用方必须明确传入 `--start/--end` 或 SDK 参数。窗口仍使用原视频秒数，不能把选中 4 秒后的结果称为完整理解了整段视频。

加上 `--live` 才会发起 **一次** DeepSeek 请求，使用 `.env` 中 Settings 的凭证、默认 `detail=low`、thinking disabled、最多 1024 输出 tokens。没有自动重试或自动提高预算：

```text
uv run --extra local --extra vision python examples/video_question.py clip.mp4 "只看13至17秒，角色按什么顺序出现？" --start 13 --end 17 --live
```

若 `requires_review=true`，示例和适配器默认拒绝付费发送。先检查警告和时间戳，缩小窗口或合理增加 `--max-frames`；只有明确接受遗漏风险时才使用 `--allow-review --live`。这会降低证据要求，不会让遗漏的事件恢复。模型回答也不会自动保存为事实。

`--detail original` 可保留选中帧的当前尺寸；选帧阶段已将 JPEG 缩至最长边不超过 1280，所以它不是原始视频像素的无损上传。更多帧和更高 detail 可能提高费用；成本以实际 usage 为准，不按 MP4 文件大小或 JPEG 压缩率估算。示例仅限制单次请求，应用循环需要独立的累计次数和费用预算。

## API 与 SDK

启动 `fishkeeping serve` 后，以下接口沿用 namespace API key 和 scope 边界：

| 接口 | 返回 | 是否保存记忆 | 是否调用模型 |
| --- | --- | --- | --- |
| `POST /v1/videos/analyze` | `VideoEvidence`，包含短期选中帧 JPEG Base64 | 否 | 否 |
| `POST /v1/videos/events` | `EventResponse`，来源 episode 与重放状态 | 仅保存来源 JSON | 否 |

请求的共同字段如下；视频事件再增加 `event_id` 和 `session_id`：

```json
{
  "scope": {"namespace": "default", "user_id": "alice", "agent_id": "default"},
  "video_base64": "<MP4/AVI Base64 without a data: prefix>",
  "question": "只看13至17秒，角色按什么顺序出现？",
  "max_frames": 6,
  "strategy": "adaptive",
  "start_seconds": 13,
  "end_seconds": 17,
  "max_chars": 6000
}
```

默认最多 6 帧，帧预算范围 2–12；近重复画面可能让实际帧数低于预算。`uniform` 为均匀候选基线，`adaptive` 组合变化与时间覆盖。`max_chars` 约束不含像素的完整来源 JSON，是字符数而非 tokens；不足时返回错误，不静默截断帧来源。`start_seconds` 默认 0，`end_seconds=null` 表示末个实际解码时间戳。

```python
from pathlib import Path
from fishkeeping.client import MemoryClient
from fishkeeping.models import Scope

video = Path("clip.mp4").read_bytes()
options = {"start_seconds": 13, "end_seconds": 17, "max_frames": 6}
with MemoryClient("http://127.0.0.1:8765", Scope(user_id="alice")) as memory:
    evidence = memory.analyze_video(video, "这段画面中的角色顺序是什么？", **options)
    print(evidence.packed)  # 无帧 Base64，适合检查来源与警告
    receipt = memory.enqueue_video(
        video, "这段画面中的角色顺序是什么？", "video-001", "inspection", **options
    )
    print(receipt.episode.id)
```

分析与首次存档是两个独立本地调用，目前会分别解码；无须存档时只调用 analyze。示例可用 `--url http://127.0.0.1:8765` 改走 API，另加 `--event-id video-001` 才保存来源。启用认证时，应用应向 `MemoryClient` 提供其 namespace 对应的 API key；不要把服务 key 或视频字节写进应用日志。

## 证据、时间与回放

`VideoEvidence` 分开报告全片的实际末个 PTS、容器声明时长、解码帧数、候选数，以及分析窗口、窗口候选数、选中帧、最大时间间隙、检测变化与被保留变化数量。每帧记录真实 decoder 时间戳、原视频帧索引、图片 SHA256、尺寸和选择原因；原始视频另有 SHA256。相同画面再次出现可以保留重复图片哈希，帧 ID/索引/时间戳保持严格递增。

`packed` 含这些来源与限制，但没有视频字节或帧图片。可靠事件保存为 tool 消息，`extract=false`，不自动把图中文字或模型推断提取为用户事实。相同原始视频、问题、scope、时间窗与预算参数重放同一 event ID，读取已保存的来源，不再次解码；参数改变返回 409。删除来源后保留墓碑，拒绝重放，已确认事实若通过 `source_ids` 引用了 episode，会继续使用现有级联删除约定。

**事件回放不能恢复像素或重新观看视频。** 调用方须保留原始媒体并核对哈希，需要新的视觉分析时再次调用 analyze。事件来源也不是完整的语义转录，不能把帧选择元数据当作已经识别出的剧情。模型回答如需进入记忆，应先确认内容并显式引用源 episode。

## 资源与错误边界

本地上限：24 MiB 输入、180 秒声明时长、6000 解码帧、12M 单帧像素、最长边 8192；最多 90 候选帧，每张候选 JPEG ≤256 KiB。支持 MP4-family/AVI，拒绝 URL、播放列表及无法获得有效时间元数据的文件。超出上限应由调用方显式裁剪/转换后提交，服务不会自动绕过预算。

解码在带 45 秒墙钟超时的独立子进程进行，临时文件随后清理。每进程一次解码，繁忙返回 429；非法输入、异常 PTS、明显提前 EOF 或窗口不足两候选返回 422；缺少可选依赖返回 503。容器时长与帧数允许小幅误差，具体容差见管线文档。解码预算不是经过生产压测的并发容量，也不是操作系统级内存沙箱。

服务账户必须能在系统临时目录创建、写入和清理文件。若受限 Windows 环境报告 `temporary storage or decoder process permission denied`，可在启动服务或离线示例前把 `TEMP` 和 `TMP` 指向已存在、可写的临时目录；不要关闭输入校验或改为远程解码。该错误不代表视频容器已损坏。

## 如何解释本轮测试

本轮视频用例取自有明确许可的 Big Buck Bunny 动画预告片，两个问题来自同一约 32 秒来源，不能算作两个独立视频，更不能冒充复杂实拍视频验证。标签与评分时间区间先于模型调用冻结；评分区间不传给选帧器，只有问题已明确指定的用户窗口可作为请求参数。

完整视频均匀取帧和显式短窗口自适应取帧有不同覆盖范围；比较 token 时必须同时说明范围、帧数、detail 和遗漏，不只报告下降比例。关键词匹配不能证明事件先后顺序，未覆盖的事件也不能仅凭故事常识补全。本轮两题输入约减少 59%，但第二题优化输出校验失败，未重试；同窗六帧消融不支持 adaptive 优于 uniform。实际成本、人工内容核验和失败案例见 [真实素材评测](media-evaluation.md) 与 [验收记录](verification.md)。

当前限制仍包括：无音频，抽帧间的短事件可能遗漏，细小文字与轻微运动可能丢失，不证明持续动作、因果关系或某事件没有发生。`requires_review=false` 也不是任务正确率保证。后续需要增加不同来源的有许可实拍视频、音画不同步/快动作、复杂镜头和长视频评测。
