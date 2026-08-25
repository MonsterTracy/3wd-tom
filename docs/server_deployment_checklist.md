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
3. 完整测试在本地提交前执行；服务器更新后只核对精确 commit、配置和采集入口，不重复扫描带软链接的数据目录或运行完整测试集。
4. 三局诊断使用 `configs/twd_tom_server_qwen35_9b.yaml`（seeds 4101--4103，target 3）；60 局正式采集使用 `configs/twd_tom_server_qwen35_9b_canonical_60.yaml`（预声明 seeds 4201--4320，target 60，game-level split 48/6/6）。CLI 必须分别使用 `--game-count 3` 或 `--game-count 120`，并与所选配置的 `pipeline.collection` 完全相同。
5. 为每次尝试选择全新的 `run_id` 和不存在的输出目录，例如 `/data/yuxiao/3wd-tom/canonical_data/<run_id>`。
6. 先用 `--mode pilot` 跑小批诊断。每次 backend 瞬时异常最多执行 3 次显式、计数的尝试；每个 gameplay 生成阶段、PRE readonly label 和独立 speech parser 的完整生成都最多 3 次。公开发言 realization 与 speech parser 的后续尝试只携带上一次验证错误并完整重生成，不修补或部分接受旧响应。SDK 内部 retry 保持为 0，不启用 provider fallback 或动态 replacement seed。
7. label 请求使用按 observer/hard knowledge 动态收窄的 JSON Schema，直接排除 observer 自身和已知非狼人，并令 `minItems` 等于必须包含的已知其他狼人数；具体成员与重复项仍由本地 parser 严格验证（不向 vLLM xgrammar 发送其不支持的 `uniqueItems`、`contains` 或 `minContains`）。后续 label 尝试只附加上一次本地验证错误并重新生成完整 JSON。每次原始响应、状态与错误都写入 `call_audit.json`，但不会删除非法成员、补猜或伪造标签。
8. pilot 在 gameplay 三次生成都失败后可以提交确定性的合法 fallback action；若某条 PRE snapshot 的 label 三次仍失败，则跳过整条 snapshot，并由当前 speaker 提交固定 `no_commitment` 发言继续。speech parser 三次仍失败时保留 `status=error` 与所有 attempts 并继续后续局数，不伪造 action。整批始终不能物化为 canonical 数据。
9. 正式采集必须显式使用 `--mode canonical`。canonical 允许上述有界尝试，但三次 gameplay、label、realization 或 speech parser 生成仍失败时终止当前局，将其原样移入 `failures/`，然后继续下一个预声明种子。失败局永不提交 fallback action、不接受缺失 PRE snapshot，也不进入成功 game 集合。
10. 逐局确认存在 `speech_annotations.jsonl`；canonical 的每条公开发言都必须是 `annotation_source=llm_parser` 且不能是 `status=error`。pilot 可出现显式 error 用于诊断。生成器 intent 不能替代独立 parser 标注。
11. 正式 60 局批次只在 `completed_game_count=60` 且 `target_reached=true` 时获得 canonical 资格；预声明池中允许存在被完整审计的失败种子，但这些失败不能进入 `games/`、split 或训练数据。若 120 个种子耗尽仍不足 60 局，批次为 incomplete，不得物化。
12. 采集完成后运行 `python -m script.twd_tom.audit_canonical_belief_data --canonical-root ...`；该命令会验证成功 batch 的 plan/summary/per-game/failure digest、种子分区、完整 PRE 覆盖与 snapshot SHA256 链，并拒绝 pilot、未达 target、成功局中的 label failure 或任何 gameplay fallback。只有 audit `status=PASS` 才能物化数据集。
13. 进程或 SSH 会话中断后，可对同一输出目录使用完全相同的 commit、配置、`run_id`、mode、seed start 和 game count 加 `--resume`。已成功或已失败的种子一律跳过；中断时留下的半局会被记为 `interrupted_previous_process` 失败且不会重跑。协议、commit 或 seed plan 变化时必须新建 `run_id`，不能混入旧批次。

## DeepSeek 发言解析器影子审计

该步骤只用于比较不同模型对同一批公开发言的三元组解析，不是 canonical 采集、标签修补或数据物化。输入批次保持只读，输出必须位于独立的 `review` 目录。

1. 在服务器 shell 中设置 `DEEPSEEK_API_KEY`，不要把密钥写入仓库、命令历史或输出工件。
2. 使用 `configs/twd_tom_shadow_deepseek_v4_flash.yaml`。请求固定为 `temperature=0`、thinking disabled、最多 3 次完整生成；SDK retry 为 0，不接受部分响应或 fallback action。
3. 对已完成 pilot 执行：

```bash
cd /home/dell/yuxiao/3wd-tom

SOURCE_ROOT="/data/yuxiao/3wd-tom/canonical_data/<pilot_run_id>"
SHADOW_ROOT="/data/yuxiao/3wd-tom/review/<pilot_run_id>_deepseek_v4_flash"

test -f "${SOURCE_ROOT}/summary.json"
test ! -e "${SHADOW_ROOT}"

/home/dell/yuxiao/envs/3wd-tom/bin/python \
  -m script.twd_tom.audit_shadow_speech_parser \
  --input-root "${SOURCE_ROOT}" \
  --config configs/twd_tom_shadow_deepseek_v4_flash.yaml \
  --output-root "${SHADOW_ROOT}" \
  --env-file /home/dell/yuxiao/3wd-tom/.env
```

4. 审阅 `${SHADOW_ROOT}/summary.json` 的 parser error、retry、exact-order match、action-set match 和 disagreement 计数，再查看逐局 `shadow_speech_comparisons.jsonl`。shadow 输出永不被 canonical validator、materializer 或训练入口读取。

## 数据物化、训练与评估

1. 用 `script.twd_tom.materialize_canonical_belief_dataset` 按 game-level split 直接物化 raw belief snapshots；输出目录必须同时包含 `split_manifest.json`，当前主线没有 C1、D 或 `split_offline_d_training_data` 阶段。不要手工移动、改名或拼接三份 JSONL。
2. `train` 只接受同一个 `split_manifest.json` 指定的 train/validation；test 只由 `eval` 读取。`eval` 要求 checkpoint 的 manifest digest 与 test 所属 manifest 一致，并验证 test 与 train、validation 均无 game overlap。
3. 数据集写入 `/data/yuxiao/3wd-tom/datasets`，checkpoint 和 metrics 写入 `/data/yuxiao/3wd-tom/outputs`。
4. 保存 commit、配置 SHA256、`run_id`、模型 ID、endpoint、seeds、环境版本、采集 plan/summary/failure、每局 call audit、数据 audit、split manifest、checkpoint 和 metrics。
5. 模型服务与长任务交给服务器 service manager 或会话管理器；仓库不负责自动下载模型或守护进程。

### 60 局 dense ToM 开发实验

1. 原 48/6/6 split 中的 test 六局保持封存。模型选择阶段只合并 train 与 validation 为 54 局开发集并生成 5-fold；fold 目录没有 `test.jsonl`，训练入口会拒绝任何含 sealed test game ID 的 fold。
2. 先分别审计原 train 与 validation 的 dense strict-PRE 契约。审计要求每个 earlier boundary 都是最终 encoded game sequence 的精确前缀；`max_seq_len=256` 截断破坏该关系时直接失败，不丢弃边界或改写数据。
3. dense 训练的 `batch_size=8` 表示 8 局；每局只运行一次 causal backbone，并在该局所有 PRE boundaries 输出监督。初始候选固定 4 层、hidden 256 的当前 Qwen2 backbone，最多 80 epochs，warmup-cosine，patience 12。
4. 先运行 Dense-A（learning rate `1e-4`、minimum `1e-5`、seed 42）。OOF 报告覆盖全部 54 局且每局只出现一次，主门槛为 observer-weighted normalized reducible-gap improvement `>=0.50`。同时比较 fold-train-only global prior 与 phase prior，并报告 game-level bootstrap 95% CI。
5. 只有 Dense-A 未达门槛时才运行 Dense-B（learning rate `3e-4`、minimum `3e-5`，其余设置完全相同）。不能根据封存 test 选择 learning rate、epoch 或 seed。
6. `run_development_oof` 会复用配置完全一致且已有成功 `summary.json` 的 fold；若某个 fold 只有不完整输出，则停止并要求检查该 fold，不删除其他已完成 fold，也不从 fold 0 重新训练。
7. 在 OOF 方案冻结并完成开发集最终拟合前，不运行 `script.twd_tom.eval`。封存 test 只允许对最终选定的单一 checkpoint 评估一次。

Dense-A 的服务器命令模板如下。路径使用项目内的 `datasets`、`outputs` 软链接，以便训练 provenance 保存 repository-relative path；物理文件仍写入 `/data/yuxiao/3wd-tom`。训练只使用项目 Python 环境，不需要启动 vLLM。

```bash
set -euo pipefail
cd /home/dell/yuxiao/3wd-tom

SOURCE_SPLIT="/home/dell/yuxiao/3wd-tom/datasets/canonical60_qwen35_cc81f96_20260825_023121_split42_48-6-6"
FOLD_ROOT="/home/dell/yuxiao/3wd-tom/datasets/canonical60_qwen35_cc81f96_20260825_023121_dev54_folds5_seed42"
AUDIT_ROOT="/data/yuxiao/3wd-tom/review/canonical60_dense_pre"

mkdir -p "${AUDIT_ROOT}"
test ! -e "${FOLD_ROOT}"
test ! -e "${AUDIT_ROOT}/train_dense_pre.json"
test ! -e "${AUDIT_ROOT}/validation_dense_pre.json"

/home/dell/yuxiao/envs/3wd-tom/bin/python \
  -m script.twd_tom.audit_dense_belief_dataset \
  --dataset "${SOURCE_SPLIT}/train.jsonl" \
  --split-name train \
  --max-seq-len 256 \
  --output "${AUDIT_ROOT}/train_dense_pre.json"

/home/dell/yuxiao/envs/3wd-tom/bin/python \
  -m script.twd_tom.audit_dense_belief_dataset \
  --dataset "${SOURCE_SPLIT}/validation.jsonl" \
  --split-name validation \
  --max-seq-len 256 \
  --output "${AUDIT_ROOT}/validation_dense_pre.json"

/home/dell/yuxiao/envs/3wd-tom/bin/python \
  -m script.twd_tom.materialize_development_folds \
  --train "${SOURCE_SPLIT}/train.jsonl" \
  --validation "${SOURCE_SPLIT}/validation.jsonl" \
  --output-dir "${FOLD_ROOT}" \
  --fold-count 5 \
  --fold-seed 42
```

确认两个 audit 均为 `PASS`，fold manifest 的 `development_game_ids` 为 54、`sealed_test_game_ids` 为 6 后，后台启动 Dense-A：

```bash
cd /home/dell/yuxiao/3wd-tom

FOLD_ROOT="/home/dell/yuxiao/3wd-tom/datasets/canonical60_qwen35_cc81f96_20260825_023121_dev54_folds5_seed42"
OOF_OUTPUT="/home/dell/yuxiao/3wd-tom/outputs/tom60_dense_a_lr1e-4_seed42_oof5"
OOF_LOG="/data/yuxiao/3wd-tom/logs/tom60_dense_a_lr1e-4_seed42_oof5.console.log"

test ! -e "${OOF_OUTPUT}"
test ! -e "${OOF_LOG}"

nohup /home/dell/yuxiao/envs/3wd-tom/bin/python \
  -m script.twd_tom.run_development_oof \
  --fold-root "${FOLD_ROOT}" \
  --output-dir "${OOF_OUTPUT}" \
  --epochs 80 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --min-learning-rate 1e-5 \
  --warmup-ratio 0.05 \
  --early-stopping-patience 12 \
  --early-stopping-min-delta 1e-4 \
  --seed 42 \
  --device cuda \
  --bootstrap-samples 2000 \
  --target-improvement 0.50 \
  > "${OOF_LOG}" 2>&1 < /dev/null &

OOF_PID=$!
echo "OOF_PID=${OOF_PID}"
echo "OOF_LOG=${OOF_LOG}"
```

用 `tail -f "${OOF_LOG}"` 查看日志；最终报告为 `${OOF_OUTPUT}/oof_summary.json`。SSH 断开不影响 `nohup` 进程。重新执行完全相同的 OOF 命令时，已完成且配置一致的 fold 会跳过；但首次命令中的 `test ! -e "${OOF_OUTPUT}"` 只用于防止误覆盖，恢复时应省略这一个检查。
