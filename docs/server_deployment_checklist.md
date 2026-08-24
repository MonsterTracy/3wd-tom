# 服务器部署与运行清单

## 固定目录职责

- Git 项目：`/home/dell/yuxiao/3wd-tom`。只保存源码、配置和文档。
- 项目环境：`/home/dell/yuxiao/envs/3wd-tom`。用于测试、采集编排、数据物化、训练和评估。
- 大模型推理环境：`/data/yuxiao/3wd-tom/envs/3wd-inference`。只用于启动 Qwen3.5 等推理服务。
- 大文件根目录：`/data/yuxiao/3wd-tom`。保存模型权重、canonical 数据、materialized datasets、日志和训练输出。

项目目录中的 `canonical_data`、`datasets`、`logs`、`models`、`outputs` 等只能是指向大文件根目录的软链接；这些名字已同时按目录和软链接加入 `.gitignore`。

## 首次按本地当前版本重建

1. 先在本地完成测试、commit 和 push；服务器只部署一个已提交的精确 commit。
2. 在服务器记录现有 `/home/dell/yuxiao/3wd-tom` 的 commit、`git status` 和软链接目标。真实目录 `data`、`logs` 若含文件，先迁移到 `/data/yuxiao/3wd-tom`，不要直接删除。
3. 旧 `.bundle` 和 `/data/yuxiao/3wd-tom/tom-bundles` 仅是历史传输介质；确认目标 commit 已存在于 GitHub 后可以删除，不作为部署来源。
4. 将旧项目目录改名留作临时回滚副本，再从 GitHub 全新 clone 到 `/home/dell/yuxiao/3wd-tom`；不要把本地完整工作树复制到服务器。
5. checkout 预定 branch 和 commit，确认 `git status --short` 为空。
6. 建立指向 `/data/yuxiao/3wd-tom` 的软链接，并再次确认 `git status --short` 仍为空。正式采集和训练都会拒绝 dirty worktree。
7. 在 `/home/dell/yuxiao/envs/3wd-tom` 中执行 `python -m pip install -e /home/dell/yuxiao/3wd-tom`。
8. 只在服务器创建 `.env`；不要提交或从工作站传输已有 `.env`。

## 正式采集前

1. 在 `/data/yuxiao/3wd-tom/envs/3wd-inference` 启动 OpenAI-compatible Qwen3.5 服务；模型权重保持在 `/data/yuxiao/3wd-tom/models`。
2. 验证 `http://127.0.0.1:8000/v1/models`，再发出一次有 token 上限的 `/v1/chat/completions` 请求。
3. 使用项目环境运行 `python -m pytest -q`。
4. 使用 `configs/twd_tom_server_qwen35_9b.yaml`；CLI 的 seed 范围和 game count 必须与其中 `pipeline.collection` 完全相同。
5. 为每次尝试选择全新的 `run_id` 和不存在的输出目录，例如 `/data/yuxiao/3wd-tom/canonical_data/<run_id>`。
6. 先用 `--mode pilot` 跑小批诊断。每次 backend 瞬时异常最多执行 3 次显式、计数的尝试；每个 gameplay 生成阶段遇到截断、JSON/动作解析或发言质量失败时最多生成 3 次；PRE readonly label 遇到解析或语义失败时也最多生成 3 次。SDK 内部 retry 保持为 0，不启用 provider fallback 或 replacement seed。
7. label 请求使用按 observer/hard knowledge 动态收窄的 JSON Schema，直接排除 observer 自身和已知非狼人；重复项由本地 parser 拒绝（不向 vLLM xgrammar 发送其不支持的 `uniqueItems`）。每次 label 尝试的原始响应、状态与错误都写入 `call_audit.json`，但不会删除非法成员、补猜或伪造标签。
8. pilot 在 gameplay 三次生成都失败后可以提交确定性的合法 fallback action；若某条 PRE snapshot 的 label 三次仍失败，则跳过整条 snapshot，并由当前 speaker 提交固定 `no_commitment` 发言继续。plan、summary 和 call audit 会记录模式、缺失 PRE、label failure 与 fallback；即使没有实际触发 fallback，整批也不能物化为 canonical 数据。
9. 正式采集必须显式使用 `--mode canonical`。canonical 允许上述有界尝试，但三次 gameplay 或 label 生成仍失败时立即终止，永不提交 fallback action，也不接受缺失 PRE snapshot。
10. 逐局确认存在 `speech_annotations.jsonl`；每条公开发言都必须是 `annotation_source=llm_parser` 且不能是 `status=error`。生成器 intent 不能替代独立 parser 标注。
11. 采集完成后运行 `python -m script.twd_tom.audit_canonical_belief_data --canonical-root ...`；该命令会验证成功 batch 的 plan/summary/per-game digest、完整 PRE 覆盖与 snapshot SHA256 链，并拒绝 pilot、label failure 或任何含 fallback 的批次。只有 audit `status=PASS` 才能物化数据集。
12. 失败后保留失败工件并更换 `run_id`，不得覆盖或续写原目录。

## 数据物化、训练与评估

1. 用 `script.twd_tom.materialize_canonical_belief_dataset` 按 game-level split 直接物化 raw belief snapshots；输出目录必须同时包含 `split_manifest.json`，当前主线没有 C1、D 或 `split_offline_d_training_data` 阶段。不要手工移动、改名或拼接三份 JSONL。
2. `train` 只接受同一个 `split_manifest.json` 指定的 train/validation；test 只由 `eval` 读取。`eval` 要求 checkpoint 的 manifest digest 与 test 所属 manifest 一致，并验证 test 与 train、validation 均无 game overlap。
3. 数据集写入 `/data/yuxiao/3wd-tom/datasets`，checkpoint 和 metrics 写入 `/data/yuxiao/3wd-tom/outputs`。
4. 保存 commit、配置 SHA256、`run_id`、模型 ID、endpoint、seeds、环境版本、采集 plan/summary/failure、每局 call audit、数据 audit、split manifest、checkpoint 和 metrics。
5. 模型服务与长任务交给服务器 service manager 或会话管理器；仓库不负责自动下载模型或守护进程。
