# DeepSeek 低 token 识图调研

核实日期：2026-10-28（Asia/Shanghai）。以下为官方文档、官方模型卡、仓库及发布包的读取结果；读取资料不代表运行过相应模型。性能与费用实测另见验收记录。本文没有调用付费接口或下载 DeepSeek 权重。

## 当前托管接口确实支持视觉

当日 `deepseek-flash` 对应 **DeepSeek-V4.1-Flash-1017**，支持图像；`deepseek-v4-pro` 不支持。旧 `deepseek-v4-flash-vision-exp` 已退役并被路由到最新 Flash，不能据旧文章推断现在的能力。[官方模型与价格](https://api-docs.deepseek.com/quick_start/pricing/)

Chat Completions 的 user 消息可包含 `text` 和 `image_url` 内容块，后者接受真实图像 URL 或 `data:image/png;base64,...`。Responses API 使用 `input_image`。**把 Base64 塞进普通文本 content 不构成视觉调用**。本项目应复用现有 HTTPX/Settings，凭证只在请求头出现；图片内容和 OCR 文字均作为不可信证据。[Chat Completions 规范](https://api-docs.deepseek.com/api/create-chat-completion/)、[Responses 说明](https://api-docs.deepseek.com/guides/responses_api/)

官方 Vision 规则摘要：`detail=low` 会先缩到 512×512；`high`、`original` 和当前 `auto` 保留原图。图像最终按尺寸转换并缩放：约 544×544 面积以下放大，大图缩至约 1300×1300 面积，单图最多 1024 tokens，多张分别计算。JPEG/PNG/GIF/WebP 均支持。图像的字节数、压缩率、Base64 字符数与视觉 tokens 没有同比关系；多裁剪图的总成本可能高于一张原图。[官方 Vision](https://api-docs.deepseek.com/guides/vision/)

官方尺寸计算器也注明结果只是估计，真正计费以响应 `usage` 为准。因此实现只用单图 1024 作为保守视觉预算，并记录实际 prompt/completion/cache usage；不要把本地 OCR 字符估计或模型内部视觉 token 数冒充托管 usage。[Token & Token Usage](https://api-docs.deepseek.com/quick_start/token_usage/)

| Flash 每百万 tokens（USD） | 高峰 | 非高峰 |
| --- | ---: | ---: |
| 输入，cache miss | 0.30 | 0.15 |
| 输入，cache hit | 0.006 | 0.003 |
| 输出 | 1.20 | 0.60 |

高峰为工作日 UTC 01:00–04:00、06:00–10:00，中国法定节假日除外；其他时间为非高峰。实验预估统一使用高峰、全部输入未缓存的上界，实际价格快照保留日期。以 8 次、每次最多 2 张图（2048 视觉 tokens）、另 2000 文本 tokens、1024 输出 tokens 估算，总额约 **$0.01955**，低于 $0.05；这是事前上界示例，不是发生的费用，失败/重试仍计入 8 次硬上限。[当日官方价格](https://api-docs.deepseek.com/quick_start/pricing/)

## 官方开源权重与托管 API 是不同部署方式

以下大小来自官方 Hugging Face API 的文件元数据，单位 GB 为十进制，仅为权重文件；运行内存另需框架、激活与 KV cache，不能把下载大小当作显存要求。

| 项目 | 固定代码/模型版本 | 权重与许可 | 本阶段判断 |
| --- | --- | --- | --- |
| [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) | Git `09eaf526153e7a01ed16c9dea8c96282aaea29c0`；HF `9f30c71f441d010e5429c532364a86705536c53a` | 3,336,106,240 BF16 参数；6,672,547,120 bytes；代码/模型 MIT | 文档 OCR 模型；官方示例 NVIDIA CUDA、BF16、FlashAttention，不选为本机 CPU 默认 |
| [DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2) | Git `2f3699ebbb96fa8af32212e8c170f2cc28730fad`；HF `aaa02f3811945a91062062994c5c4a3f4c0af2b0` | 3,389,119,360 BF16 参数；6,778,573,880 bytes；代码/模型 Apache-2.0 | 动态分辨率文档识别，官方模型卡默认 `(0–6)×144+256` 视觉 tokens；这是开源模型内部编码，不是 Flash API 计费 |
| [DeepSeek-VL2-tiny](https://github.com/deepseek-ai/DeepSeek-VL2) | Git `ef9f91e2b6426536b83294c11742c27be66361b1`；HF `66c54660eae7e90c9ba259bfdf92d07d6e3ce8aa` | 3,370,501,440 BF16 总参数；6,741,334,208 bytes；代码 MIT，权重 DeepSeek License Agreement 1.0 | 视觉问答/定位候选；1B 是激活参数，不能当作总权重；官方示例 `.cuda()`，不纳入本阶段 CPU 默认 |
| [Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B) | Git Janus `1daa72fa409002d40931bd7b36a9280362469ead`；HF `960ab33191f61342a4c60ae74d8dc356a39fafcb` | `pytorch_model.bin` 4,178,890,389 bytes；代码 MIT；模型卡正文指向 DeepSeek Model License，虽然页面标签为 MIT | 通用理解/生成，理解编码器为 384×384；许可元数据与正文不一致，部署前按实际权重许可文件复核，当前不采用 |

OCR 两代官方环境为 Python 3.12.9、CUDA 11.8、torch 2.6.0、transformers 4.46.3、FlashAttention 2.7.3，示例使用 `trust_remote_code=True`。本项目没有运行这些仓库的远程代码，没有证明它们不能在 CPU 移植，但官方路线、下载量与本阶段低成本目标不匹配。[OCR 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-OCR/tree/9f30c71f441d010e5429c532364a86705536c53a)、[OCR-2 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2/tree/aaa02f3811945a91062062994c5c4a3f4c0af2b0)、[OCR-2 许可](https://github.com/deepseek-ai/DeepSeek-OCR-2/blob/2f3699ebbb96fa8af32212e8c170f2cc28730fad/LICENSE.txt)、[VL2 权重许可](https://github.com/deepseek-ai/DeepSeek-VL2/blob/ef9f91e2b6426536b83294c11742c27be66361b1/LICENSE-MODEL)、[Janus 权重许可](https://github.com/deepseek-ai/Janus/blob/1daa72fa409002d40931bd7b36a9280362469ead/LICENSE-MODEL)

## 本地 OCR 选型

选择 **`rapidocr==3.10.0` + CPU ONNX Runtime** 作为可选依赖。此版本 2026-10-08 发布，Python >=3.8,<4，Git tag 对应 `bd941dae09da793e76c6ba15c7aaf40f8424232a`。wheel 为 27,288,134 bytes，SHA-256 `2fc34e26cd0f48514804a94f8832283039c7334076bff60768b1fed9ad068050`；已下载到内存、核验 PyPI SHA 并读取归档，未执行包代码。代码 Apache-2.0；随包 `MODEL_LICENSES.md` 独立记录 PaddleOCR 权重归属和 Apache-2.0 条款。[PyPI 固定版本](https://pypi.org/project/rapidocr/3.10.0/)、[固定源码](https://github.com/RapidAI/RapidOCR/tree/bd941dae09da793e76c6ba15c7aaf40f8424232a)、[官方检测模型卡](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx)、[官方识别模型卡](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx)

| 随包默认模型 | bytes | SHA-256 |
| --- | ---: | --- |
| `PP-OCRv6_det_small.onnx` | 9,929,594 | `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f` |
| `PP-OCRv6_rec_small.onnx` | 21,234,383 | `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884` |
| `ch_ppocr_mobile_v2.0_cls_mobile.onnx` | 585,532 | `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c` |

三模型共 31,749,509 bytes。默认使用中英文识别配置；复杂表格结构、场景对象关系、颜色、计数及图片语义不能仅由 OCR 文本可靠恢复。返回的识别置信分数不是校准过的“事实为真”概率。

读取发布包确认：`RapidOCR(params={...})` 支持点分配置键；`EngineConfig.onnxruntime.intra_op_num_threads=1`、`inter_op_num_threads=1` 限线程。关闭 CUDA/DML/CANN/CoreML，显式指定 `Det/Cls/Rec.model_path` 到已核验随包文件，避免模型缺失时自动联网补下载。普通调用返回 `.boxes/.txts/.scores/.elapse`；空结果字段可为 None。`result.vis()` 会延迟取字体，因此运行服务只使用结构化结果。实例调用会更新内部配置，应由适配层串行化。以上是源码检查结论，运行时仍需离线测试。

备选比较：旧 `rapidocr-onnxruntime==1.4.4` 的 wheel 更小（14.9 MB），但 Python <3.13 且归档未附独立模型许可清单；[Tesseract](https://github.com/tesseract-ocr/tesseract) 为 Apache-2.0、CPU 方案，但需额外系统二进制和语言数据，Windows/Ubuntu 安装路径不同；[Windows OcrEngine](https://learn.microsoft.com/en-us/uwp/api/windows.media.ocr.ocrengine) 返回行/词坐标但依赖 Windows 和机器已安装语言，不能作为跨平台服务默认。此阶段不引入额外 GPU VLM。

图像解码/裁剪建议固定 `Pillow==13.0.0`（Python >=3.11，MIT-CMU）；当日 Windows CPython 3.12 wheel 7,272,166 bytes。CPU runtime 当日最新 `onnxruntime==1.31.1`（Python >=3.11，MIT）；实际项目锁文件可保留与向量后端共同验证的兼容固定版本，不因本调研强制升级现有运行时。[Pillow 元数据](https://pypi.org/pypi/Pillow/13.0.0/json)、[ONNX Runtime 元数据](https://pypi.org/pypi/onnxruntime/1.31.1/json)

## 推荐证据链与实验设计

1. 文字图片先在本地解码，限制像素/尺寸/格式，计算原图 SHA-256；相同图片、相同方法/模型版本可复用 OCR，不跨 scope 共享私人证据。
2. OCR 保留原图坐标、文字、置信、方法版本与模型 hash；按问题选完整文本块，保留邻行、表头与单位，压成紧凑结构化文本。预算不足时报告省略，不能把裁掉的内容当作不存在。
3. 非文字场景、未检出文字、低置信或结构关系问题需要视觉证据。可用一张低分辨率预览表达全局，再在必要时增加有限 ROI；只有任务允许付费且预算足够才发送。预览与 ROI 分别收费，避免无上限切块。
4. OCR 文字或图中“忽略规则”等内容是证据，不可提升为 system 指令。记忆写入应保留图像 hash、选择区域、提取器、模型版本和回答来源；用户偏好/事实不应从一张无身份确认的截图自动推定。
5. 合成截图/表格、非文字图形、低清/旋转、无答案与图中文字注入分别评测。冻结问题、必要证据和答案后运行原图视觉、低 detail/ROI、完整 OCR、按问题压缩 OCR；独立报告识别正确率、必要信息遗漏、回答正确率、tokens、调用数、峰值估算费用、墙钟/CPU/内存。

任何离线字符/token 估计必须标为估计；没有真实 API usage 的组不能填“实际 tokens”。小型合成集结果不等于真实照片、复杂账单或生产场景表现，也不能保证压缩后的答案不变。真实请求由统一有上限的实验入口发出，避免 OCR、问答、记忆提取各自重复付费。
