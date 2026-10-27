# 可恢复的本地增量索引

实施日期：2026-10-28。增量索引解决新记忆和后台新提取事实需要手工补索引的接入缺口。已有 `/v1/index` 和 `index` 命令保留，查询仍只编码问题。

## 启动和运行预算

在 `.env` 配置 `FISHKEEPING_EMBEDDING_BACKEND=local`，安装 local extra 后，针对应用使用的 scope 启动独立进程：

```text
uv run --extra local fishkeeping index-worker --user-id alice --namespace default --agent-id default --batch-size 32 --max-batches 20 --daily-limit 1000
```

默认每次最多编码 32 条且合计不超过 32000 字符；每个进程最多执行 20 个有实际尝试的批次，全库每个 UTC 日最多预留 1000 条记录尝试。只运行一次可加 `--once`，无待办时立即返回。常驻运行时，每次空转、繁忙或编码失败后按 2、4、8……60 秒退避；成功后重置退避。默认读取一个明确 scope，多个用户需要分别安排进程或调度调用。

达到进程批次上限或全库日预算后进程退出；需要持续运行的部署应由进程管理器/定时任务再次启动。日预算耗尽后不要高频重启，可在下个 UTC 日运行。命令不会自行安装系统服务。即使配置了远程向量 API，`index-worker` 也拒绝远程模式，避免后台隐式付费；手工 `index` 和 `/v1/index` 是独立显式操作，不受该 worker 的预算限制。

预算在数据库事务中先预留，再进行本地计算。失败、进程崩溃、删除竞态或租约丢失均不退还预留数；这是计算尝试上限，不是成功条数。所有 scope/model 共享同一 UTC 日预算。同一天多个进程使用不同配置时，已记录的有效日上限只能降低，不能通过重启或提高参数增加；次日按第一个运行进程的配置初始化。

## 恢复和并发语义

不另外保存业务内容副本。`memories` 中有效且缺少当前模型缓存的记录就是持久待办；旧记录优先，下一轮重新扫描，因此服务写入、可靠事件产生的 episode、worker 新提取事实、模型 key 改变都能自动被发现。已经完成的缓存行是进度检查点，重启不会重新编码它们。

同 scope/model 在 SQLite `BEGIN IMMEDIATE` 事务内取得五分钟租约；其他进程返回 busy。崩溃后租约到期可重新计算，本地 CPU 重复计算不产生 API 费用，但两次尝试都消耗预算。提交时检查租约 token 和到期时间，并在同一事务检查 memory 仍属于该 scope、active 且未过期；旧进程无法覆盖新进程结果，删除、归档、替代和过期也不能使旧缓存重新生效。不同模型 key 独立索引，不覆盖已有版本。

只为长期记忆建索引，保持现有 `embedding_missing()` 的语义；会话 working 记忆仍走已有词法路径。索引状态继续由现有覆盖率和 `hybrid_partial`/警告反映。损坏缓存仍需显式 `index --rebuild`；worker 不把存在但损坏的缓存当作“缺失”自动覆盖。

## 数据库演进和回滚

核心 schema 仍为 v2。首次启动 worker 在事务内追加 `index_worker_schema`（版本 1）、`index_worker_leases` 和 `index_worker_daily_budget` 三张表，不修改业务记录；未知 worker schema 版本拒绝启动。表创建失败会整体回滚。旧 v0.2 Store 可以重新打开该数据库并使用核心能力，但旧程序没有增量 worker。

升级前按运行文档备份。回滚应用时停止新 worker，保留这些附加表即可；若完整恢复旧备份，预算记录也回到备份时刻，应暂停 worker 到次日或保留原日预算，避免把恢复备份当作重置预算。不要通过删除附加表“修复”当天预算。旧模型缓存的磁盘清理由显式管理完成。

## 验证和已知限制

离线测试覆盖重启续作、新写入发现、跨 scope 日预算、跨日恢复、失败消耗、并发租约、过期租约 fencing、删除/归档/TTL/事实替代竞态、模型切换、字符与条数上限、working 排除、未知 schema、查询不编码文档及 CLI 空转退避。运行：

```text
.venv\Scripts\pytest.exe -q tests/test_index_worker.py tests/test_cli.py --basetemp=.tmp/pytest-index-worker
```

本机 Windows 11 / Python 3.12.4、固定 MiniLM 缓存、`HF_HUB_OFFLINE=1` 的真实本地 smoke 结果见 [index-worker-smoke.json](index-worker-smoke.json)。初始两条含模型加载 1.8583 秒；重开数据库补一条 0.0060 秒；幂等空转 0.0001 秒；后续新写入一条 0.0066 秒。它是四条合成记忆的功能 smoke，不是生产吞吐或 p95。所有生命周期检查通过，无付费调用。`Python` 直接查询正确；跨语言改写 `What programming language does the user use?` 在现有 0.50 阈值下返回空，原始失败保留，没有调低阈值或修改记忆制造通过。

增量构建不改变召回算法。上一阶段万条 384 维精确扫描约一秒的合成诊断表明大库仍有扫描成本；本阶段优先补齐接入和可恢复进度，不新增 ANN 依赖。下一阶段应以真实库规模、并发与延迟目标决定 ANN/PostgreSQL 迁移，并保留词法与精确扫描降级。
