# Agent 记忆系统：GitHub 调研与选型依据

核实日期：**2026-10-28（Asia/Shanghai）**。目标：构建可由其他 agent 接入、能够持久保存经验、事实和技能、后续接入 DeepSeek 的独立记忆服务。

## 调研方法与边界

通过 GitHub 连接器读取下列仓库的 README、根目录 LICENSE 和 `main` 提交信息；对 Generative Agents 另读检索与反思源码，对 LangMem 另读概念文档。使用公开网页搜索补充论文入口。以下是静态调研，没有安装、运行或复现这些上游项目的性能；不把上游宣传、托管平台成绩当作本项目性能。许可证名称来自实际读取的文件，项目内子目录与第三方资源仍可能有独立许可。

此次仅参考机制，自行实现 Fishkeeping 核心，不直接复制上游代码。选择 Python + FastAPI + SQLite 的目的是降低首次接入成本，并保证没有大模型 API key 时也能完成写入、检索和持久化；这属于本项目的工程判断。

## 仓库对照

| 项目与一手资料 | 实际核实的许可 | 核心价值 | 对 Fishkeeping 的决定 |
| --- | --- | --- | --- |
| [mem0ai/mem0](https://github.com/mem0ai/mem0) · [README](https://github.com/mem0ai/mem0/blob/main/README.md) | [Apache-2.0](https://github.com/mem0ai/mem0/blob/main/LICENSE) | 用户、会话、agent 维度的记忆接口；抽取与多信号检索 | 借鉴接口与隔离维度；不作为必须在线调用模型的核心依赖 |
| [letta-ai/letta](https://github.com/letta-ai/letta) → [letta-ai/letta-code](https://github.com/letta-ai/letta-code) | 两仓库均为 [Apache-2.0](https://github.com/letta-ai/letta-code/blob/main/LICENSE)；[旧仓库许可](https://github.com/letta-ai/letta/blob/main/LICENSE) | 有状态 agent、记忆块、会话搜索、反思 | 借鉴工作上下文和长期记忆的分离；不引入完整 agent 运行时 |
| [getzep/graphiti](https://github.com/getzep/graphiti) · [README](https://github.com/getzep/graphiti/blob/main/README.md) | [Apache-2.0](https://github.com/getzep/graphiti/blob/main/LICENSE) | 时间图谱、持续摄入、语义/关键词/图遍历混合检索 | 借鉴来源、时间与关系建模；图数据库留作后续可替换后端 |
| [langchain-ai/langmem](https://github.com/langchain-ai/langmem) · [概念文档](https://github.com/langchain-ai/langmem/blob/main/docs/docs/concepts/conceptual_guide.md) | [MIT](https://github.com/langchain-ai/langmem/blob/main/LICENSE) | 语义、情景、程序性记忆；即时写入与后台巩固 | 借鉴类型划分和纯转换逻辑；不要求接入方使用 LangGraph |
| [joonspk-research/generative_agents](https://github.com/joonspk-research/generative_agents) · [论文](https://arxiv.org/abs/2304.03442) | [Apache-2.0](https://github.com/joonspk-research/generative_agents/blob/main/LICENSE) | 记忆流、相关性/新近性/重要性评分、带证据的反思 | 借鉴可解释排序与经验巩固；不搬入模拟小镇和旧运行环境 |
| [MemTensor/MemOS](https://github.com/MemTensor/MemOS) · [README](https://github.com/MemTensor/MemOS/blob/main/README.md) | [Apache-2.0](https://github.com/MemTensor/MemOS/blob/main/LICENSE) | 统一记忆 API、记忆空间、纠错；本地插件与完整服务两条路线 | 借鉴本地持久化与存储隔离；暂不绑定插件宿主或完整服务基础设施 |

## 逐项核实与适用性

### 1. Mem0：记忆层 API 与当前版本的变化

当前 README 已介绍 2026 年的追加式抽取、多信号检索及时间推理；不能沿用旧资料，把更新/删除判定描述为现版唯一算法。README 也明确区分开源 SDK 和含专有优化的托管平台，平台成绩不可直接外推到自部署。库默认使用在线 LLM 与 embedding，支持替换提供商。

Fishkeeping 借鉴 `add/search` 使用方式和作用域过滤。基础读写不依赖模型，模型抽取是可选增强；保留原始事件和变更来源，避免无法追查的静默覆盖。来源：[当前 README](https://github.com/mem0ai/mem0/blob/main/README.md)。

### 2. Letta / MemGPT：需要区分历史与现行代码

`letta-ai/letta` 的当前 README 明确说明：旧 V1 API 服务位于 `archive` 分支，现行开发在 `letta-ai/letta-code`。现行项目包含 agent harness、CLI、App Server、记忆块、git 跟踪的 MemFS 和反思能力，属于完整 agent 产品。不能按过去 MemGPT 的安装说明，假定旧 Python 服务仍是当前主线。

Fishkeeping 借鉴“有限工作上下文 + 持久记忆”的分层，但仅提供记忆能力，不接管用户未来的 DeepSeek agent 执行循环。来源：[迁移入口](https://github.com/letta-ai/letta/blob/main/README.md)、[现行项目](https://github.com/letta-ai/letta-code/blob/main/README.md)。

### 3. Graphiti：时间、实体和关系

Graphiti 是时间上下文图框架。README 列出 Neo4j、FalkorDB、Amazon Neptune 等后端，以及 LLM/embedding 配置；提供参考 REST 和 MCP 服务。新信息持续进入图谱，检索结合语义、关键词和关系。其 Kuzu 驱动已被 README 标记弃用，选型时不宜依据旧教程选它。

Fishkeeping 首版以 SQLite 中的记忆及来源关系满足审计和演进需求；只有出现跨实体多跳查询、规模或并发要求时，再评估图后端。不能把一个关系表宣称为已实现完整 Graphiti 时间图谱。来源：[README](https://github.com/getzep/graphiti/blob/main/README.md)。

### 4. LangMem：记忆类别与形成时机

LangMem 区分事实知识、过去经验和行为程序；区分对话中及时形成记忆与对话后的后台巩固。其核心转换函数可与不同存储组合，较高层工具与 LangGraph Store 集成。README 的 `InMemoryStore` 示例明确不跨进程重启保存，生产持久化需数据库后端。

Fishkeeping 采用这些类别，并将“抽取候选记忆”和“数据库事务写入”分开，以便测试和更换模型。来源：[概念文档](https://github.com/langchain-ai/langmem/blob/main/docs/docs/concepts/conceptual_guide.md)、[README](https://github.com/langchain-ai/langmem/blob/main/README.md)。

### 5. Generative Agents：排序与反思的原始实现

实际读取的 `retrieve.py` 将新近性、重要性和 embedding 相关性组合，召回后更新访问时间；`reflect.py` 将洞见与证据节点关联，并根据累计重要性触发反思。源码参数和实现细节不宜直接视为通用最优值。README 的测试环境是 Python 3.9.12，项目同时包含环境服务器和模拟逻辑。

Fishkeeping 借鉴多因素排序、访问强化、证据关联；遗忘采用可解释的工程衰减，而非声称复现人脑机制。来源：[检索源码](https://github.com/joonspk-research/generative_agents/blob/main/reverie/backend_server/persona/cognitive_modules/retrieve.py)、[反思源码](https://github.com/joonspk-research/generative_agents/blob/main/reverie/backend_server/persona/cognitive_modules/reflect.py)。

### 6. MemOS：完整服务与本地插件要分别比较

README 展示完整自部署服务依赖 Neo4j + Qdrant；另有本地 SQLite 插件，提供 FTS5 + 向量混合检索、经验/策略分层等能力。当前已列出 DeepSeek Harness 接入，但这代表特定 agent 宿主插件适配，不能据此推断任意 DeepSeek API 集成无需开发。

Fishkeeping 借鉴本地 SQLite 路线和统一记忆接口；本项目希望服务于任意 HTTP 客户端，暂不继承插件宿主约束。来源：[README](https://github.com/MemTensor/MemOS/blob/main/README.md)、[本地插件目录](https://github.com/MemTensor/MemOS/tree/main/apps/memos-local-plugin)。

## 落到 Fishkeeping 的设计原则

1. **记忆分层**：工作记忆服务当前会话；情景记忆保存经历；语义记忆保存稳定事实；程序性记忆保存可复用步骤。分类是系统约定，不代表神经科学的完整实现。
2. **先保存证据，再做概括**：巩固结果关联原始记忆；模型推断与用户明确提供的信息可区分，错误概括可以追溯。
3. **召回可解释**：相关性是主要因素，新近性、重要性和强化参与排序；无关内容不能仅因“重要”进入查询结果。
4. **遗忘不是误删**：时间衰减、过期、归档和用户明确删除是不同操作；默认检索排除过期或被替代的信息，删除行为需明确。
5. **模型与存储解耦**：无 key 时核心可用；DeepSeek 用于可选抽取/巩固/对话，embedding 通过独立适配器接入，不假设对话模型同时提供 embedding。
6. **接入独立于框架**：HTTP/OpenAPI 为主，Python 客户端为便捷入口；调用方不必迁移到任何上游 agent 框架。
7. **作用域始终随请求校验**：项目、用户、agent、会话隔离不能只在提示词中约定；检索、关系、巩固和删除都要使用相同边界。

以上为调研形成的设计方向，实际实现范围、接口、测试结果以后续架构与运行文档为准。首版不宣称达到上述项目的性能、完整时间图谱或生产集群能力。

## 可复核的代码快照

以下是同次调研读取到的 `main` 提交；不是安装版本锁定。后续需要复制或改造上游源码时，应固定提交、重新核实许可并保留适用声明。

| 仓库 | 观察到的提交（可点击复核） |
| --- | --- |
| mem0ai/mem0 | [26cdefb46600c0fa5953c4b5ee5003c125539fc4](https://github.com/mem0ai/mem0/commit/26cdefb46600c0fa5953c4b5ee5003c125539fc4) |
| letta-ai/letta | [8816d3a8516893bfdc0fa14bfb1235226012e295](https://github.com/letta-ai/letta/commit/8816d3a8516893bfdc0fa14bfb1235226012e295) |
| letta-ai/letta-code | [4c78cbe420d00beac1be5d63c4f7b99e7cd9554b](https://github.com/letta-ai/letta-code/commit/4c78cbe420d00beac1be5d63c4f7b99e7cd9554b) |
| getzep/graphiti | [2b5db9e139b2627c3ba18ff55f9181569b0d8224](https://github.com/getzep/graphiti/commit/2b5db9e139b2627c3ba18ff55f9181569b0d8224) |
| langchain-ai/langmem | [48e3c11f5bb527282c7d5339c6a87a0b35abccfc](https://github.com/langchain-ai/langmem/commit/48e3c11f5bb527282c7d5339c6a87a0b35abccfc) |
| joonspk-research/generative_agents | [fe05a71d3e4ed7d10bf68aa4eda6dd995ec070f4](https://github.com/joonspk-research/generative_agents/commit/fe05a71d3e4ed7d10bf68aa4eda6dd995ec070f4) |
| MemTensor/MemOS | [adcb9f270c982f2b9ff8db9714792829d3bd6ab8](https://github.com/MemTensor/MemOS/commit/adcb9f270c982f2b9ff8db9714792829d3bd6ab8) |

## 补充论文与后续评估

- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)：记忆流与反思的研究依据；论文关注模拟行为可信度，不能等同于通用记忆服务正确率。
- [A Machine with Short-Term, Episodic, and Semantic Memory Systems](https://ojs.aaai.org/index.php/AAAI/article/view/25075)：在特定强化学习环境研究分层记忆；支持分类思路，不证明本项目实现具有相同能力。

首版验收重点应是跨重启持久化、中文与英文召回、作用域隔离、过期排除、重复请求、冲突替代、巩固可追溯、提供商失败处理。长期质量评估需固定数据集、模型版本、成本和延迟条件，分别测量检索与最终回答，不能直接比较不同厂商的宣传分数。
