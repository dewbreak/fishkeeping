# DeepSeek 低 token 图片问答

日期：2026-10-28，v0.3。官方托管 `deepseek-flash` 当前支持原生视觉，`deepseek-v4-pro` 不支持；模型版本、官方接口、开源 OCR 权重与许可见 [调研](research/deepseek-vision.md)。本项目使用标准 `image_url` 内容块，Base64 只是图片传输格式，不作为普通文字让模型“猜图”。

## 选择路径

| 输入 | 默认处理 | 需要回退的情况 |
| --- | --- | --- |
| 文字截图、印刷文档 | CPU OCR → 问题词匹配 → 相邻行/同一行单元格 → 紧凑 JSON | 未命中、低置信、预算省略、细字、复杂版面 |
| 简单表格 | 保留目标行附近和同一水平带的单元格与坐标 | 合并单元格、复杂表头、跨页关联、图表数值推断 |
| 非文字场景 | 显式 DeepSeek 原生视觉 `low` | 细节不足时显式改用原图或区域 |

OCR 不理解鱼类、病变、对象关系或图表趋势。`task="scene"` 不运行 OCR，返回 `requires_review=true`；不会偷偷调用视觉模型。`requires_review=false` 只表示当前启发式没有触发回退，不保证所有所问字段都已覆盖。词法筛选不保证跨语言/同义词匹配，复杂问题应使用区域或原图；置信值不是校准后的正确概率。

## 安装和接口

```text
uv sync --extra dev --extra local --extra vision --locked
uv run --extra local --extra vision fishkeeping serve
```

`vision` 固定 RapidOCR 3.10.0（Apache-2.0）、ONNX Runtime 1.31.1、Pillow 13.0.0。随包三个 ONNX 模型共 31.75 MB，校验 SHA256 后使用明确本地路径，CPU 两线程、不执行远程代码、不自动下载模型。基础服务无需安装 vision；调用图片功能缺少依赖时返回 503。普通 CI 仅安装 Pillow 生成测试图片，OCR 使用 fake，不下载权重。

`POST /v1/images/analyze` 使用与其他 API 相同的 namespace 认证与 scope。请求：

```json
{
  "scope": {"namespace": "default", "user_id": "alice", "agent_id": "default"},
  "image_base64": "<PNG/JPEG base64, without data: prefix>",
  "question": "For tank B2, what are the pH and nitrate values?",
  "task": "document",
  "max_chars": 3000,
  "region": null
}
```

限制：单图 4 MiB、12M 像素、最长边 8192；仅 PNG/JPEG。先应用 EXIF 方向，坐标均相对于旋转后的显示图，`region={"x1":0,"y1":0,"x2":800,"y2":600}`。OCR 预览最长边 1800，最多保留 256 个识别区域参与筛选，超出明确标记遗漏。`max_chars` 范围 800–10000，是包含来源信息的 JSON 字符预算，不是 tokens。过长问题与来源超出预算返回 422，不截断来源或半条证据。

响应 `ImageEvidence` 包含 `image_sha256`（原始文件字节）、宽高、方法与版本、问题、选中区域 ID/坐标/文字/置信、总区域数/省略数、警告、`requires_review` 和 `packed`。`packed` 可作为不可信参考发给文本模型。API 不读任意路径/URL，不将原图写入数据库；调用方应保存原图与哈希以便复查。

重复图片只在进程内去重，缓存最多 8 份文本，按 scope、图片、方法版本和区域隔离。不同问题复用 OCR 后重新筛选。缓存有界且不保存图片字节；删除记忆清除对应 scope 的缓存。

## 可靠图片事件与来源

`POST /v1/images/events` 在上述请求增加 `event_id` 和 `session_id`，返回 202 及 `{evidence,event}`。CPU OCR 完成后，证据通过原有可靠事件事务保存；不生成付费提取任务。持久事件使用 tool 角色，图中文字不能变成 user 指令或直接成为用户事实。原图不落库，所以首次 OCR 完成前的进程失败仍需调用方重发。

相同 scope/event ID/图片/问题/区域/task/预算/session 重试读取保存的证据，重启或 OCR 升级不重新识别。修改任何原始请求参数返回 409；需要重新分析用新 event ID。删除源 episode 后，墓碑拒绝重放，所有引用它的事实按完整来源图级联删除。保存证据并不代表验证了图片内容。

```python
from pathlib import Path
from fishkeeping.client import MemoryClient
from fishkeeping.models import Scope

with MemoryClient("http://127.0.0.1:8765", Scope(user_id="alice")) as memory:
    result = memory.enqueue_image(
        Path("panel.png").read_bytes(), "Filter price?", "panel-001", "inspection"
    )
    print(result.evidence.packed)
    # 仅在应用或人工确认后保存事实，并保留图片证据来源。
    # memory.remember("Filter price: 25 USD", kind="semantic",
    #                 source_ids=[result.event.episode.id])
```

## 显式模型调用

```python
from fishkeeping.config import Settings
from fishkeeping.image_chat import DeepSeekImageChat
from fishkeeping.vision_models import Box

chat = DeepSeekImageChat(Settings())
try:
    answer = chat.answer_evidence(result.evidence)  # review=true 时拒绝自动发送
    # 以下每次也是独立付费请求，只选择真正需要的一条路径：
    # answer = chat.answer_image(image_bytes, question, detail="low")
    # answer = chat.answer_image(image_bytes, question, detail="low",
    #     regions=[Box(x1=100, y1=200, x2=700, y2=450)])
    print(answer.answer, answer.usage)
finally:
    chat.close()
```

每次最多 1024 输出 tokens、thinking disabled、无重试。原生视觉仅允许已核实的 Flash，发送一张预览与最多两张原尺寸区域。区域不是必然省钱：每张图独立计费，预览+区域可能比单张原图更多。`answer_evidence(..., allow_review=True)` 是调用方明确接受低置信证据的实验选项，不建议自动打开。

示例 [image_question.py](../examples/image_question.py) 默认只做本地处理；`--live` 才执行一次请求。所有图中文字与 OCR 都由 system 规则声明为不可信数据；这减少指令混淆，但不宣称彻底解决提示注入。回答也不自动保存为事实。

## 实测对比

[冻结图片集](../tests/fixtures/image_cases.json) 由本项目原创合成，无私人业务内容，标签 SHA256 `f1a41e44fa9aeb4ba149c6bf02f4695db516fd100beaa7fbf18691100c5fd7c1`。图片生成到 `.tmp/image-evaluation`，不提交二进制素材；不同操作系统字体可影响 OCR，报告记录实际字体。离线 8 案例含长截图、表格、中文、倒置、低清、无答案、场景、注入文字。7 张文字图本机 CER 为 0，8/8 所需证据短语保留；无答案/场景明确要求回退。这些容易的合成图不能代表照片或真实复杂文档质量。

| 案例/路径 | 实际输入 tokens | 实际输出 tokens | 请求秒数 | 答案词组检查 |
| --- | ---: | ---: | ---: | --- |
| 长截图原图 | 1116 | 22 | 1.1113 | 通过 |
| 长截图紧凑 OCR | 366 | 22 | 0.7253 | 通过 |
| 表格原图 | 692 | 38 | 0.6522 | 通过 |
| 表格紧凑 OCR | 458 | 46 | 0.8273 | 通过 |
| 简单场景原图 | 689 | 28 | 0.9473 | 通过 |
| 简单场景 low | 309 | 31 | 0.6789 | 通过 |

输入分别减少 **67.2%、33.8%、55.2%**。6 次共 3630 输入、187 输出 tokens，按峰值未缓存单价估算 **US$0.0013134**，不是账户账单；评分为冻结词组+`uncertain=false`，不是人工完整语义评测。模型响应不写入报告，记录 usage/耗时/匹配结果和图片哈希，见 [真实记录](validation/images-live-2026-10-28.json)。

表中延迟仅是模型 HTTP 请求，不含 OCR；[离线报告](validation/images-offline-2026-10-28.json) 单独记录墙钟、CPU 时间、进程累计峰值 RSS 与缓存重放。峰值约 581 MiB，包含模型加载和图片；不是单次增量内存或并发容量。完整 OCR 原文字数与紧凑正文分别为长截图 708→138、表格 93→46，但带来源 JSON 为 632/695 字符，短文本时来源结构开销可能超过纯全文 OCR。不能把字符下降当作 token 等比例下降，更不能用文件压缩率推断视觉 tokens。

## 复验与限制

```text
python scripts/evaluate_images.py --ocr --output .tmp/images-offline.json
python scripts/live_images.py --preview --report .tmp/images-preview.json
python scripts/smoke.py --local-embeddings --vision
```

使用已安装的 `.venv`，或 `uv run --extra dev --extra local --extra vision --locked`。`--preview` 无网络和付费；`live_images.py --live` 会计费，并与 `live_task_answers.py` 共用持久 8 次/US$0.05 账本。本阶段已完成 6 图片+2记忆请求，不再调用。失败/未知结果先扣预算，重启不清零，不自动重试；未来新阶段需要新的明确实验预算。

未做真实照片、复杂表格、视频、多页扫描、OCR/VLM 本机 GPU 对比或图像向量召回。原生裁剪请求结构、坐标和上限已离线验证，本轮未额外花费真实 ROI 请求。下一阶段优先采集有许可的真实困难样本，量化完整任务准确率与证据遗漏，再考虑更复杂版面/VLM。
