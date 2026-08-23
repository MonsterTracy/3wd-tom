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
6. 运行 `python -m script.twd_tom.collect_canonical_trajectories ...`。不启用自动重试、replacement seed 或 provider fallback。
7. 采集完成后运行 `python -m script.twd_tom.audit_canonical_belief_data --canonical-root ...`；只有 audit `status=PASS` 才能物化数据集。
8. 失败后保留失败工件并更换 `run_id`，不得覆盖或续写原目录。

## 数据物化、训练与评估

1. 用 `script.twd_tom.materialize_canonical_belief_dataset` 按 game-level split 直接物化 raw belief snapshots；当前主线没有 C1、D 或 `split_offline_d_training_data` 阶段。
2. `train` 只读取 train/validation；test 只由 `eval` 读取。
3. 数据集写入 `/data/yuxiao/3wd-tom/datasets`，checkpoint 和 metrics 写入 `/data/yuxiao/3wd-tom/outputs`。
4. 保存 commit、配置 SHA256、`run_id`、模型 ID、endpoint、seeds、环境版本、采集 plan/summary/failure、每局 call audit、数据 audit、split manifest、checkpoint 和 metrics。
5. 模型服务与长任务交给服务器 service manager 或会话管理器；仓库不负责自动下载模型或守护进程。
