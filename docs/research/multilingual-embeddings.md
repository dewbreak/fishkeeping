# 本地多语言向量选型与接入约定

调研日期：2026-10-28。本文以官方仓库、模型卡和发布者文件为依据；模型质量、速度和平台可用性必须以本项目真实验收结果为准。

## 决策

采用可选依赖 `fastembed>=0.9.1,<0.10`，通过 ONNX Runtime 在 CPU 上运行 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。默认维持原有远程向量配置行为；显式选择 `embedding_backend=local` 才启用本地后端。FastEmbed 当前 PyPI 版本为 0.9.1，官方库采用 Apache-2.0；不需要部署 Qdrant 服务，也不引入 PyTorch 推理环境。[FastEmbed 官方仓库](https://github.com/qdrant/fastembed)、[发布元数据](https://pypi.org/pypi/fastembed/json)、[官方依赖定义](https://github.com/qdrant/fastembed/blob/v0.9.1/pyproject.toml)。

本地推理不产生按次模型 API 费用；首次启用需要下载模型，之后使用缓存。CPU 时间、内存、磁盘和下载流量仍有成本。DeepSeek 继续负责可选事实提取，两者职责独立。

## 比较与证据

| 选项 | 语言与维度 | 许可证 | 输入约定 | 当前接入判断 |
| --- | --- | --- | --- | --- |
| paraphrase-multilingual-MiniLM-L12-v2 | 约 50 种语言，含英语、简体/繁体中文；384 维 | Apache-2.0 | 句子相似度模型；query/document 均使用原文；推荐上下文 128 token | FastEmbed 官方内建模型，本轮采用 |
| multilingual-e5-small | 约 100 种语言，含中英文；384 维 | MIT | 检索时分别加 `query: ` 和 `passage: `；非英文同样需要；512 token | 目前不是 FastEmbed 内建 registry 项，需要自定义注册、固定导出与独立评估 |

来源：[MiniLM 官方模型卡](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/blob/main/README.md)、[E5 官方模型卡](https://huggingface.co/intfloat/multilingual-e5-small/blob/main/README.md)、[FastEmbed 内建模型定义](https://github.com/qdrant/fastembed/blob/v0.9.1/fastembed/text/pooled_embedding.py)、[FastEmbed 模型列表](https://qdrant.github.io/fastembed/examples/Supported_Models/)。这里将 FastEmbed 的实际 registry 与示例中的自定义模型区分开；出现示例名称不代表内建支持。

E5 发布者提供约 470 MB 的标准 ONNX、235 MB 的 O4 ONNX，以及约 118 MB 的 `model_qint8_avx512_vnni.onnx`。这些文件的精度和优化目标不同，不能把任意小体积文件当成通用 CPU 版本。本轮先使用已经由 FastEmbed 注册、具有明确 tokenizer/pooling 配置的 MiniLM；今后如切换 E5，要同步更改前缀、缓存命名和验收基准。[E5 发布者 ONNX 文件清单](https://huggingface.co/api/models/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3/onnx)。

## 固定模型文件，避免缓存漂移

本轮实际使用的 ONNX 导出由 Qdrant 发布：

| 项目 | 固定值 |
| --- | --- |
| 用户配置模型名 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 文件仓库 | `Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q` |
| snapshot revision | `faf4aa4225822f3bc6376869cb1164e8e3feedd0` |
| 权重文件 | `model_optimized.onnx`，235,052,644 bytes |
| 必需文件 | 权重、`config.json`、`tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json` |
| 选定文件总大小 | 约 252.14 MB；不含 Python 包、缓存额外开销 |
| 运行参数 | `CPUExecutionProvider`，默认 2 线程，推理批次 32，不启用多进程并行 |

**文件仓库名中的 `-Q` 不等于 INT8。** 官方 `ort_config.json` 显示 `fp16: true`，而 `quantization` 为空；本项目记录为 FP16 优化 ONNX，不宣称 INT8，也不推断在所有 CPU 上都节省一半内存或运行更快。该导出的配置包含 GPU 优化选项，故必须完成实际 CPU 推理验收。[固定版本文件清单](https://huggingface.co/api/models/Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q/tree/faf4aa4225822f3bc6376869cb1164e8e3feedd0)、[导出配置](https://huggingface.co/Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q/blob/faf4aa4225822f3bc6376869cb1164e8e3feedd0/ort_config.json)。

适配器使用 Hugging Face `snapshot_download` 的固定 revision 和文件白名单，先只查本地缓存，缺失时才下载；随后将固定目录传给 FastEmbed 的 `specific_model_path`，并设 `local_files_only=True`。不会加载仓库内 Python 代码，不使用 `trust_remote_code`。只接受本轮审核的模型名，未知模型返回受控错误。[FastEmbed 路径加载接口](https://github.com/qdrant/fastembed/blob/v0.9.1/fastembed/text/text_embedding.py)、[Hugging Face 下载接口](https://huggingface.co/docs/huggingface_hub/package_reference/file_download#huggingface_hub.snapshot_download)。

`model_key` 同时包含 FastEmbed 安装版本、原始模型名、固定 snapshot revision、FP16/mean/L2/384 输出约定，以及 `token126-weighted-chunks-v1`。模型文件、后处理或分段方式变化必须生成新 key，并重建相关向量。远程和本地模型的向量不能混用。

## 128 token 限制与长文本处理

MiniLM 的模型卡给出 128 token 的句子编码长度，Qdrant tokenizer 配置也同时存在 `max_length=128` 与 `model_max_length=512`。FastEmbed 0.9.1 取二者较小值。因此直接把 4,000 或 16,000 字符的记忆交给模型会丢失尾部；不能仅凭底层 512 位置容量推断完整输入已被编码。[固定 tokenizer 配置](https://huggingface.co/Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q/blob/faf4aa4225822f3bc6376869cb1164e8e3feedd0/tokenizer_config.json)、[FastEmbed tokenizer 实现](https://github.com/qdrant/fastembed/blob/v0.9.1/fastembed/common/preprocessor_utils.py)。

本项目适配器执行以下约定：

1. 用同一固定 snapshot 的独立 tokenizer 读取全文 offset，关闭该 tokenizer 的截断与 padding；不修改模型推理用的 tokenizer。
2. 正文每段最多 126 token，给 CLS/SEP 预留 2 token。按字符 offset 切片，保证拼接所有段后等于原文。
3. 对每个切片重新分词校验。遇到多 token 共用一个 Unicode span、字节回退或切片后 token 数变化时，缩小切片并重试，不能依赖原始 offset 推断新 token 数。
4. 短文本只推理一次；长文本各段先 L2 归一化，再按实际正文 token 数做加权均值，最后再次 L2 归一化。查询和文档执行同一分段约定，分别调用 `query_embed` / `passage_embed`，不人为添加 E5 前缀。
5. 每个文本最多 16,000 字符；每批最多 1,024 个文本、262,144 字符、2,048 个片段。超过限制返回 `invalid_request`，调用方应减小后台索引批次。

服务层显式索引每个推理批次进一步限制为最多 32 条记忆、累计 32,000 字符；单条合法记忆最多 16,000 字符，因此长记忆每批最多两条。`IndexRequest.limit` 仍限制本次请求处理的总记录数，各批执行前仍检查时间预算，已经写入的向量可以被下次请求复用。字符数不能严格等价于 token 数：极端 tokenizer 展开仍可能触发适配器的片段总量限制，此时返回受控 `invalid_request`，不静默截断内容，也不递归进行隐式重试。

均值融合保留尾部参与计算，但不能保证长篇多主题记忆的单点信息仍可被高分召回。上游仍应优先提取短而完整的原子事实；未来可再评估逐段存储、多向量检索。这里不把长文本聚合向量等同于原生长上下文模型。

## 接口、错误与生命周期

实现位置：[`local_embeddings.py`](../../src/fishkeeping/local_embeddings.py)。

```python
provider = LocalEmbeddingProvider(settings)
document_vectors = provider.embed(["我喜欢乌龙茶。"])
query_vector = provider.embed_query("oolong beverage")
provider.close()
```

模块导入、对象初始化以及空文档批次都不加载模型、不下载文件。第一次显式索引或查询时才加载；同一实例复用模型。加载、tokenizer、生成器消费、推理、关闭均受同一锁保护。不同进程各自持有 ONNX session，因此增加 API worker 数会增加常驻内存。

返回向量必须严格为 384 维、数值有限、非零，并与输入数量相符；结果统一为 Python 浮点列表。错误统一映射为 `ProviderError`，不暴露源文本、缓存路径、异常正文。没有安装可选包时提示 `pip install '.[local]'`；加载失败可在下一次显式调用时重试。

缓存默认是工作目录下 `.cache/embeddings`，已由 Git 忽略。多实例部署应明确配置同一缓存路径并预先准备 snapshot，避免每个实例重复下载。

在首次导入 FastEmbed/Hugging Face 前，适配器默认将 `HF_HOME` 和 `HF_XET_CACHE` 分别设为配置缓存绝对路径下的 `hub` 与 `xet`，使辅助下载缓存及 Xet 日志也落在工作区。显式环境变量优先；若进程此前已导入这些库，应在启动前配置环境变量并重启，避免库沿用导入时的缓存常量。

## 验收与费用控制

不下载模型的单元测试覆盖中英文原文路由、尾部保留、token 权重、Unicode 重叠 offset、生成器消费期间的并发互斥、加载一次后复用、固定 revision/CPU 参数、缓存优先、缺失依赖、错误输出、加载错误脱敏和关闭。测试文件：[`test_local_embeddings.py`](../../tests/test_local_embeddings.py)。

真实验收应在可靠摄入和候选检索落地后进行，明确记录模型 snapshot、平台、包版本、下载大小、索引耗时、查询耗时和测试集结果。重点检查原来的 `quiet workspace` 和跨语言 `oolong beverage` 漏召回，同时保留无关查询、过期、被替代记录、scope/session 隔离的负例。

初始余弦阈值 0.35 不能从模型卡推导为通用阈值。本轮已经完成真实缓存模型的相似度诊断，推荐本地后端使用保守的 0.50，并保留配置入口；远程模型不因此共享同一阈值。真实下载/CPU 运行及系统级评估的最终结果由验收文档记录；本研究记录不把 fake 单元测试当成模型质量证明。

## 2026-10-28 阈值诊断与回归集

[`diagnose_embeddings.py`](../../scripts/diagnose_embeddings.py) 只加载已有固定 snapshot 缓存，强制 Hugging Face 离线；不下载另一份模型，也不调用付费 API。报告保存了每题所有候选的原始 cosine、预期/非预期标签、scope/expiry/session/fact 过滤情况及词法交集，便于区分模型相似度问题与候选隔离问题。[完整诊断报告](../validation/embedding-calibration-2026-10-28.json)。

原 26 条记忆、22 题及标签没有修改。首次相似度诊断前另行编写 8 条不同主题记忆、12 个正例（跨语言翻译、同义改写、信息查询）和 12 个完全无关的问题；随后将文字、标签、顺序原样导出到 [`multilingual_holdout.json`](../../tests/fixtures/multilingual_holdout.json)。诊断脚本直接读取该文件并记录文件 SHA-256，不再保留一份可发生漂移的内嵌题库。

**这个新增集合已参与阈值选择和英文功能词问题定位，现在是固定回归/校准集，不能再声称是未见测试集，也不能据此估计生产准确率。** 文件名保留 `holdout` 以便追踪最初来源；后续扩大质量评估应另建未参与调参的数据集。

初次诊断结果如下。表内正例指标是按原始 cosine 排序后的纯语义 Recall@3；与综合词法、重要性、遗忘权重的完整服务指标不同。

| cosine 阈值 | 新增正例命中 | 新增无关问题纯语义空结果 | 原 8 个负例纯语义空结果 |
| --- | --- | --- | --- |
| 0.35 | 12/12 | 11/12 | 4/8 |
| 0.45 | 11/12 | 12/12 | 8/8 |
| 0.50 | 11/12 | 12/12 | 8/8 |
| 0.55 | 11/12 | 12/12 | 8/8 |
| 0.60 | 10/12 | 12/12 | 8/8 |

选择 0.50 是保守默认值：相较 0.45，它还能排除原集中 `timezone Shanghai` 与“回复请使用中文解释”的非预期关联（cosine 约 0.471）；0.60 则额外丢失一条有效跨语言翻译。该结论只支持当前有限回归样本中的取舍，不证明最优阈值。

保留以下已知失败，不改题目或标签使测试看起来通过：

- 原题 `oolong beverage` 与“我喜欢乌龙茶。”的正确 cosine 约 0.404985，低于原负例 `Oslo` 与当前 `Berlin` 事实的 0.412114。一个全局阈值无法同时保住前者、拒绝后者。
- 新题 `How do I travel to work?` 与“我每天骑自行车上班。”的正确 cosine 约 0.368566，保守阈值会漏掉它。
- 初次诊断中，两个新增无关英文问题分别通过 `is`、`are` 产生词法误召回；纯语义阈值再高也无法修复这一类问题。需由词法查询处理单独解决，并在完整服务层重跑同一固定集合。

完整服务回归需同时报告正例 Recall/MRR、负例空结果以及所有遗留失败；不能仅发布“语义空结果 12/12”。下一步更大的独立数据集应包含短查询、跨语言问句、专名、否定、时间关系与相近但错误的事实。
