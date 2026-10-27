# DeepSeek 与基础技术调研

核实日期：2026-10-28。资料来自官方文档和上游仓库，未运行上游系统，也未测量其宣传的性能。

## DeepSeek

- [官方 API 入口](https://api-docs.deepseek.com/)：兼容 OpenAI 风格的 Chat Completions；本项目使用 HTTPX 调用 `https://api.deepseek.com/chat/completions`，避免把记忆服务绑到某个 agent 框架。
- [JSON Output](https://api-docs.deepseek.com/guides/json_mode/)：使用 `response_format.type=json_object`，提示词必须要求 JSON 并给出结构；仍要处理空内容、截断和结构不合法。
- [接口规范](https://api-docs.deepseek.com/api/create-chat-completion/)：检查完成状态，再校验提取结果。模型名通过环境变量配置；调研当日官方入门文档使用 `deepseek-flash`，不能把历史名称长期写死。
- 模型调用是可选项；没有 key 时可完成直接写入、检索、上下文组装、生命周期管理。模型负责候选事实提取和反思，不负责数据库权限、删除或执行工具。
- 本次读取的聊天接口文档不作为 embeddings 接口存在的依据。向量服务单独配置；无向量服务时使用明确标注的词法检索，不把哈希向量宣称为语义理解。

## 记忆机制与实现依据

- [Generative Agents](https://github.com/joonspk-research/generative_agents)：观察、记忆检索、反思的研究方向；借鉴相关性、近期性、重要性的组合，不移植其模拟环境。
- [A-Mem](https://github.com/WujiangXu/A-mem)：记忆关联与演化的研究方向。首版以显式来源链接和事实版本替代推断出的全量知识图谱。
- [Python sqlite3](https://docs.python.org/3/library/sqlite3.html)：无需额外数据库服务，事务与本地文件持久化适合单机首版。
- [FastAPI 组织方式](https://fastapi.tiangolo.com/tutorial/bigger-applications/)：提供类型化 HTTP API 与自动 OpenAPI，方便其他语言和 agent 接入。

这里的工作记忆、情景记忆、语义记忆、程序性记忆是工程分类。遗忘曲线是可解释的排序启发式，不宣称重现人的生物记忆或具有经过验证的认知能力。更多仓库比较见 [GitHub 调研](github-projects.md)。
