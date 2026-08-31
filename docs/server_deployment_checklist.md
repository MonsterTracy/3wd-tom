# 服务器部署与运行清单

本清单只描述 ONUW-style ToM 的唯一正式主线：7 人、多日、纯文本普通狼人杀，public-only 模型，所有存活且标签可观测的 observer 参与监督。multimodal、private-conditioned、Annotation V2、role-scoped supervision 与 role sidecar 均不属于本清单。

## 固定目录职责

- Git 项目：`/home/dell/yuxiao/3wd-tom`，只保存源码、配置和文档。
- 项目环境：`/home/dell/yuxiao/envs/3wd-tom`，用于采集编排、数据物化、训练和评估。
- 大模型推理环境：`/data/yuxiao/3wd-tom/envs/3wd-inference`，只用于启动纯文本推理服务。
- 大文件根目录：`/data/yuxiao/3wd-tom`，保存模型权重、canonical 数据、datasets、日志和训练输出。

项目目录中的 `canonical_data`、`datasets`、`logs`、`models`、`outputs` 只能是指向大文件根目录的软链接。正式采集、final fit 与 sealed evaluation 都要求精确 commit 和 clean worktree。

## 正式任务不变量

```text
formal_supervision_mask
= observer_alive_mask & label_observed_mask
```

真实角色不得决定 formal loss population。存活且标签可观测的狼人 observer 必须进入训练、validation 与 sealed evaluation。正式 OOF、final fit、sealed eval 都不接受或读取 role sidecar。

每个 speaker 的因果链只能是：

```text
PRE readonly self-report
→ SpeakerPreSpeechBelief
├─→ supervision label
└─→ day cognition public decision
   → speech realization
   → public event
   → future PRE
```

day cognition 响应只包含 `public_content_selection`、`public_vote_stance_index`、`evidence_selection`，不生成 `belief`、`concise` 或 `roles`。

## 部署与正式采集前检查

1. 本地完成测试、人工 diff review、commit 和 push；服务器部署一个精确 commit。
2. 服务器 checkout 该 commit，并确认 `git status --short` 为空。
3. 建立大文件目录软链接，再确认 worktree 仍为空。
4. 在 `/home/dell/yuxiao/envs/3wd-tom` 执行 `python -m pip install -e /home/dell/yuxiao/3wd-tom`。
5. 只在服务器创建 `.env`，不要提交或复制工作站密钥文件。
6. 启动 OpenAI-compatible 纯文本模型服务，验证 `/v1/models` 和一次有 token 上限的 `/v1/chat/completions`。
7. 先用现有三局配置 `configs/twd_tom_server_qwen35_9b.yaml` 和显式 `--mode canonical` 启动独立的 contract-validation run；必须使用全新 validation `run_id` 和不存在的输出目录。
8. 只有该三局 canonical validation 确认 fail-closed、gameplay fallback 计数为 0、无缺失 PRE、无 parser error 并通过 canonical audit 后，才能用 `configs/twd_tom_server_qwen35_9b_canonical_60.yaml` 和另一个全新 `run_id`/输出目录启动 60 局正式 run。pilot mode 仍可作为独立 diagnostic，但不能作为 canonical collection contract 已验证的证据。

三局 canonical contract validation 使用现有 collector：

```bash
cd /home/dell/yuxiao/3wd-tom

VALIDATION_RUN_ID="<new_v6_canonical_validation_run_id>"
VALIDATION_ROOT="/data/yuxiao/3wd-tom/canonical_data/${VALIDATION_RUN_ID}"

test ! -e "${VALIDATION_ROOT}"

/home/dell/yuxiao/envs/3wd-tom/bin/python \
  -m script.twd_tom.collect_canonical_trajectories \
  --config configs/twd_tom_server_qwen35_9b.yaml \
  --run-id "${VALIDATION_RUN_ID}" \
  --seed-start 4101 \
  --game-count 3 \
  --output-root "${VALIDATION_ROOT}" \
  --mode canonical

/home/dell/yuxiao/envs/3wd-tom/bin/python \
  -m script.twd_tom.audit_canonical_belief_data \
  --canonical-root "${VALIDATION_ROOT}"
```

当前 raw schema 为 `classic7_pre_speech_player_suspicion_v6`。旧 canonical trajectory 由第二份 day-cognition belief 影响公开行为，不满足新的单一 PRE belief 因果契约，不能继续作为正式数据；必须使用新 `run_id` 重新采集。不得原地改写旧批次。

canonical 批次只有在目标成功局数达到 60、完整 PRE snapshot 与 speech annotation 审计均通过后才能物化。失败局不得进入 `games/`、split 或训练数据。采集完成后运行：

```bash
cd /home/dell/yuxiao/3wd-tom

/home/dell/yuxiao/envs/3wd-tom/bin/python \
  -m script.twd_tom.audit_canonical_belief_data \
  --canonical-root /home/dell/yuxiao/3wd-tom/canonical_data/<new_v6_run_id>
```

## 数据物化与开发集 folds

使用 game-level 48/6/6 split；test 六局保持封存。正式模型选择只合并 train 与 validation 为 54 局开发集，并生成 5-fold。不要手工移动、拼接或改名 JSONL。

```bash
set -euo pipefail
cd /home/dell/yuxiao/3wd-tom

CANONICAL_ROOT="/home/dell/yuxiao/3wd-tom/canonical_data/<new_v6_run_id>"
SOURCE_SPLIT="/home/dell/yuxiao/3wd-tom/datasets/<new_v6_split>"
FOLD_ROOT="/home/dell/yuxiao/3wd-tom/datasets/<new_v6_dev54_folds5>"

test ! -e "${SOURCE_SPLIT}"
test ! -e "${FOLD_ROOT}"

/home/dell/yuxiao/envs/3wd-tom/bin/python \
  -m script.twd_tom.materialize_canonical_belief_dataset \
  --canonical-root "${CANONICAL_ROOT}" \
  --output-dir "${SOURCE_SPLIT}" \
  --split-seed 42 \
  --train-game-count 48 \
  --validation-game-count 6 \
  --test-game-count 6

/home/dell/yuxiao/envs/3wd-tom/bin/python \
  -m script.twd_tom.materialize_development_folds \
  --train "${SOURCE_SPLIT}/train.jsonl" \
  --validation "${SOURCE_SPLIT}/validation.jsonl" \
  --output-dir "${FOLD_ROOT}" \
  --fold-count 5 \
  --fold-seed 42
```

确认 fold manifest 声明 54 个 development games、6 个 sealed games，且 fold 目录没有 `test.jsonl`。

## 正式 public-only all-alive OOF

`script.twd_tom.run_development_oof` 已固定 Qwen2、`no_phase_day`、dense supervision、public-only、V1 speech、`v1_empty_unobserved` labels 与 `all_alive`。CLI 不提供 role sidecar、supervision scope、private conditioning 或 Annotation V2 开关。

```bash
cd /home/dell/yuxiao/3wd-tom

FOLD_ROOT="/home/dell/yuxiao/3wd-tom/datasets/<new_v6_dev54_folds5>"
OOF_OUTPUT="/home/dell/yuxiao/3wd-tom/outputs/<new_v6_all_alive_oof>"
OOF_LOG="/data/yuxiao/3wd-tom/logs/<new_v6_all_alive_oof>.console.log"

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
  --reference-improvement 0.50 \
  > "${OOF_LOG}" 2>&1 < /dev/null &
```

OOF summary schema 为 `classic7_tom_v2_dense_oof_summary_v6`。旧 non-wolf OOF best epochs 及其 median 不得复用。

## Final fit 与 sealed evaluation

本轮修复后，`FinalFitConfig.epochs` 与 `EXPECTED_FOLD_BEST_EPOCHS` 保持未冻结状态；`run_final_fit` 会 fail-closed。只有新的 all-alive OOF 完成并经人工审阅后，才能在单独修改中冻结五折 best epochs 与 median epoch，再运行 final fit。final fit CLI 不接受 role sidecar。

旧 `development_final_fit_v1` checkpoint、旧 protocol digest、checkpoint SHA、Git commit 与 sealed epoch 全部失效。当前 checkpoint/protocol contract 已升级为 `development_final_fit_v2` / `classic7_tom_v2_final_fit_protocol_v2`，sealed evaluator 的新绑定常量为空并在打开 sealed label 前 fail-closed。不得在新 all-alive final artifact 审阅并冻结前运行 sealed evaluation。

## 明确排除的诊断路径

`run_diagnostic_oof`、role sidecar、`non_wolf_alive`、`villager_alive`、private-conditioned、Annotation V2、memorization sanity 与 shadow parser 可以继续作为独立 diagnostics 存在，但其输出不能进入正式 OOF summary、final-fit epoch selection、final checkpoint 或 sealed evaluation，也不能写入本正式部署清单的执行链。
