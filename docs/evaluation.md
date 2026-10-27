# 检索诊断与本地多语言评测

日期：2026-10-28。此工具用于比较代码修改前后的行为，**不是正式 benchmark，也不代表真实用户任务成功率**。数据全部为人工构造，不读取用户数据库或 `.env`。默认词法运行不发网络请求；显式 `--local-embeddings` 首次下载固定模型，之后在本地 CPU 编码，没有付费 API 调用。

## v0.3 独立任务与最终回答

新增 [冻结任务集说明](research/task-evaluation.md)：8 条时间线、21 次写入、25 查询，先冻结标签再比较，无调参。覆盖偏好更新、无 key 矛盾、错误提取和纠正、问句改写、跨语言、空结果、删除与过期。词法 Recall@3 为 0.625，本地混合为 0.90625；严格空结果均为 9/9，scope/生命周期禁止命中均为 0，已知错误高置信提取均被召回一次。原文证据包代理准确率 0.72→0.88，该代理不等同于 LLM 回答。

另用两次真实 Flash 调用进行 8 题最终回答子集对照，每条时间线只选一题。使用已保存检索结果，任务和证据 ID 改为中性编号，避免 `expense_wrong` 一类人工标签泄露；模型看不到预期答案。无证据/低置信要求严格 `INSUFFICIENT_EVIDENCE`，其他答案按冻结词组及 uncertainty 检查。词法 **5/8**，混合 **7/8**；混合失败是错误提取与原始证据冲突时模型保留不确定性，不是服务已经自动修复了事实。

两次 usage 分别为 489/179、599/182 输入/输出 tokens，模型请求耗时 1.3198/1.1098 秒，总峰值费用估算 US$0.0007596，见 [最终回答原始指标](validation/task-answers-live-2026-10-28.json)。这是共享 batch 的小型诊断，即使选取不同场景也不等同于 16 次完全独立请求；不外推完整 25 题或真实业务准确率。脚本为 [live_task_answers.py](../scripts/live_task_answers.py)，与图片验收共用已用完的 8 次账本。重启遇到已尝试阶段会保守停止，不清账本重试。

本次回归原始两套已用过的数据，Recall@3 仍为 0.928571 / 0.916667，空结果仍为 8/8、12/12；它们继续是回归集，不重新称作未见测试集。图片的识别、证据遗漏、最终回答和真实 token 比较单独见 [图片报告](vision.md)。

## 如何运行

在项目根目录执行：

```text
uv run python scripts/evaluate.py --output .tmp/evaluation.json
```

默认 `k=3`；可通过 `--k 5` 改变截断位置，或通过 `--fixture` 指定其他人工标注文件。JSON 输出包含每个案例的预期记录、实际返回、缺失记录、违规命中和检索模式。终端仅打印聚合指标。严格空结果或禁用记录检查失败时，进程非零退出；相关度指标不足不会伪装成程序异常。

固定数据位于 [recall_cases.json](../tests/fixtures/recall_cases.json)，包含 26 条记忆和 22 个查询，其中 14 个有相关标签、8 个要求严格为空。记忆涵盖中文偏好、英文实体、中文紧邻英文技术名、事件、程序步骤、会话工作记忆、过期、事实替换及三个 scope 维度。

评测使用真实 `Store` 和 `MemoryService`，独立内存数据库，模型与 embedding 显式关闭。存储及排序时钟固定为 `2026-10-28T00:00:00+00:00`，每条记忆的创建时间按固定顺序设置；这样后续日期重跑不会因 TTL 和衰减产生漂移。输出用人工 key 替代随机数据库 ID，便于版本比较。

## 指标定义

- `Recall@k`：每个有相关标签的查询中，前 k 条命中的相关标签数 / 该查询全部相关标签数，再对查询做宏平均。
- `MRR@k`：每个有相关标签的查询中，第一个相关结果的排名倒数；前 k 条未命中记为 0，再做宏平均。
- `strict_empty_passed / strict_empty_cases`：预期完全无结果的查询，有多少实际返回空列表。
- `forbidden_hits`：所有查询中明确禁止出现的记录命中数，例如其他用户的记录或已经过期、被替代的记录。

空结果查询不进入 Recall/MRR 分母，避免把“没有正例”错误计为 100% 召回。没有计算通用 Precision；除明确标注的相关与禁止记录外，其他返回条目未做完整的相关度分级。

## 当前离线结果

2026-10-28 在词法模式下实际执行默认诊断：

| 项目 | 结果 |
| --- | --- |
| 记忆 / 查询 | 26 / 22 |
| 正例查询 | 14 |
| Recall@3 | 0.857143 |
| MRR@3 | 0.857143 |
| 严格空结果 | 8 / 8 |
| 禁止记录命中 | 0 |
| 模型调用 | 0 |

12 个词法可匹配的正例在前 3 条中完整命中；两个刻意保留的语义困难例未命中：

- 查询 `quiet workspace`，目标记忆是 `I prefer a silent office.`，没有共享词项。
- 查询 `oolong beverage`，目标记忆是 `我喜欢乌龙茶。`，需要跨语言语义匹配。

这两项失败保留在原始词法基线中，没有修改查询或标签。当前词法结果保存为 [lexical-2026-10-28.json](validation/lexical-2026-10-28.json)。

## 本轮真实本地向量结果

采用 FastEmbed 0.9.1、固定 revision 的多语言 MiniLM，384 维、FP16 ONNX、CPU 两线程。模型来源、许可和分段约定见 [选型记录](research/multilingual-embeddings.md)。

| 同一原始 22 题 | Recall@3 | MRR@3 | 严格空结果 | 禁止命中 |
| --- | --- | --- | --- | --- |
| 词法 | 0.857143 | 0.857143 | 8/8 | 0 |
| 初始本地向量，阈值 0.35 | 1.000000 | 1.000000 | **4/8（失败）** | 0 |
| 最终混合检索，阈值 0.50 | **0.928571** | **0.928571** | **8/8** | **0** |

保留 [初始失败记录](validation/multilingual-initial-2026-10-28.json)、[相似度诊断](validation/embedding-calibration-2026-10-28.json) 和 [最终完整流水线记录](validation/multilingual-2026-10-28.json)。降低阈值虽然能找回 `oolong beverage`，也会把无关问题关联到已有城市等事实；其正确匹配余弦约 0.405，比一个无关匹配约 0.412 还低。因此使用保守阈值 0.50，保留这项漏召回，没有为追求 100% 而放宽无关结果门槛。`quiet workspace` 已找回对应 silent office。

另增加 [8 条记忆、24 个查询](../tests/fixtures/multilingual_holdout.json)，覆盖翻译、改写、问句及 12 个无关问题。真实完整服务得到 **Recall@3 / MRR@3 = 0.916667，严格空结果 12/12，禁止命中 0**，见 [新增集结果](validation/multilingual-regression-2026-10-28.json)。保留 `How do I travel to work?` 的漏召回。该集合最初独立构造，但已用于阈值比较和定位英文 is/are 等虚词误召回，现属于回归校准数据，不能再作为未见测试集。

词法路径也修复了仅因英文功能词重合便召回无关记忆的问题；FTS 候选和最终排序使用相同查询词过滤，保留否定、时间关系词及技术名。存储原文与索引文档分词不变。该启发式仍可能影响仅由功能词组成的查询，并不能解决所有自然语言歧义。

复验（安装 `local` extra，约 252 MB 模型下载；报告中的计时包含加载和推理，不是纯查询 SLA）：

```text
uv sync --extra dev --extra local --locked
uv run --extra local python scripts/evaluate.py --local-embeddings --output .tmp/multilingual.json
uv run --extra local python scripts/evaluate.py --local-embeddings --fixture tests/fixtures/multilingual_holdout.json --output .tmp/multilingual-regression.json
uv run --extra local python scripts/diagnose_embeddings.py --output .tmp/calibration.json
uv run --extra local python scripts/smoke.py --local-embeddings
```

`diagnose_embeddings.py` 只使用缓存，缓存缺失会失败；它输出原始 cosine 与门控诊断，不能替代 `evaluate.py` 的完整排序结果。普通 CI 不下载模型，只运行无网络适配器测试与词法诊断。

## 离线规模诊断

[万条记录报告](validation/storage-scale-2026-10-28.json) 使用 10000 条本 scope 记录、3 条其他 scope 记录、384 维固定合成向量。旧 needle 记忆能被词法与向量找回，5 次测量的词法中位数约 **0.218 ms**，完整向量 top-100 中位数约 **1.026 s**。向量由固定公式产生，不是模型语义质量测试。

环境为 Windows 11 / Python 3.12.4 / SQLite 3.45.3、内存数据库、单线程；不含磁盘 fsync、并发和模型查询编码，不是生产 p95 或 SLA。当前向量仍为 O(N×D) 精确扫描，top-k 限制不减少扫描量；规模增长后应基于业务负载引入 ANN 或专用向量索引。

```text
uv run python scripts/benchmark_storage.py --records 10000 --output .tmp/storage-scale.json
```

## 使用与扩充

每次修改检索算法时保留历史标签并比较逐例结果，再按明确需求增加匿名化案例。涉及时间、scope 和删除的检查应作为强约束；语义指标应结合真实任务的人工评估。不要为了数值提升修改预期标签，也不要把小型合成集作为生产规模或安全性证明。
