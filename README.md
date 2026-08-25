# 3WD-ToM

3WD-ToM 是一个面向经典七人狼人杀的 agent-centric Theory of Mind 研究项目。

当前游戏规则基线固定为 2 狼人、1 预言家、1 女巫、3 村民：无存活狼人时好人胜；存活狼人数达到或超过其他存活玩家总数时狼人胜。

当前 tom-v2 的基础研究问题是：给定时间点 `t` 之前已经完成的公开历史和目标观察者 `i`，预测该观察者在其他玩家之间分配的相对狼人怀疑质量：

```text
f(completed_public_history < t, observer_id=i) -> B_t(i, j)
```

`B_t(i, :)` 是由怀疑支持集归一化得到、行和为 1 的 `relative_suspicion_matrix_v1`，不是七个彼此独立且经过校准的“是狼人概率”。这一区分尤其重要：经典局有两名狼人，当前 label 只表达 agent 报告的相对怀疑支持，不提供完整角色联合分布。

## 当前唯一标签来源

训练标签只来自 playing agent 的 readonly self-report。系统在公开发言前冻结合法时间边界，对每个存活玩家调用其私有 belief query：

```text
你当前怀疑哪些玩家是狼人？
```

原始标签保存为 `suspected_werewolves` 符号集合。该集合必须包含观察者已知的其他狼人，且不得包含观察者自身或已知非狼人。请求使用按 observer/hard knowledge 动态收窄的 JSON Schema；重复项由本地 parser 拒绝。解析或语义失败最多重新生成 3 次并逐次审计；后续尝试只附加上一次本地验证错误并生成完整新响应，不修改旧响应、删除非法成员或补猜。canonical 在三次仍失败时终止当前局、保存失败审计并继续下一个预声明种子；失败局不进入 canonical 数据。pilot 可跳过整条 PRE snapshot 并以固定 `no_commitment` 发言继续，但始终不能进入训练。采集阶段不生成概率分布、belief matrix、pair target 或 21 类投影。

## 当前数据流

```text
Classic-7 simulator
  -> 发言前冻结 public history
  -> playing-agent readonly private belief query
  -> 同一份 speaker PRE belief 只读交给紧随其后的 day cognition
  -> 冻结公开表达意图并由第二次 LLM 调用生成中文为主的自然发言
  -> 独立 speech parser 只从公开原文做最多 3 次完整严格解析
  -> raw belief snapshot JSONL
  -> canonical game-level train/validation/test materialization
  -> deterministic sparse suspicion-set conversion
  -> 7×7 observer-conditioned relative-suspicion Dataset
  -> game-level dense strict-PRE Dataset
  -> observer-query belief backbone at every PRE boundary
  -> development-only 5-fold OOF training/evaluation
```

public-only reporter、external offline annotation、PBM、ToM1/ToM2 双任务和旧 formal reporter 不属于 tom-v2 主线。

公开事件与发言标注采用分层契约：`public_events` 的 `public_speech` 只保存 `event_idx`、`speaker` 和自然语言 `raw_text`；人工定义的 14 类语义动作、解析器版本、逐次原始响应和错误状态单独保存到 `speech_annotations.jsonl`。两者通过 `event_idx`、speaker 和原文 SHA256 完整绑定。canonical 数据要求每条公开发言都由独立 parser 成功标注；pilot 可保留显式 `status=error` 继续诊断，但整批永不可物化。不允许把生成器的冻结 intent 直接当成训练输入。

raw snapshot 为了证明采样边界，会保留当前 speaker 的末尾 `turn_start`；Dataset 张量化时只删除这个调度标记，模型看到的是该发言发生前已经完成的公开事件。train、validation、test 必须来自同一个经过 digest 和文件 SHA256 校验的 `split_manifest.json`；evaluation 只接受该 manifest 的 test split，并同时与 train、validation 保持 game-level 不相交。

## 代码位置

- `run_random.py`：游戏循环与唯一 belief collector 接入点。
- `werewolf/speech/private_belief_perceiver.py`：playing-agent readonly self-report。
- `werewolf/models/twd_tom/belief_snapshot.py`：逐观察者只读状态检查。
- `werewolf/models/twd_tom/samples.py`：冻结公开时间边界并生成 raw sample。
- `werewolf/models/twd_tom/public_events.py`：v4 无损公开事件序列与 raw-text-free 模型投影。
- `werewolf/models/twd_tom/speech_annotations.py`：版本化发言语义 sidecar、原文 digest 和完整绑定校验。
- `werewolf/speech/speech_perceiver.py`：独立 14-action 公开发言解析器。
- `werewolf/models/twd_tom/collector.py`：写入 raw JSONL。
- `werewolf/models/twd_tom/belief_labels.py`：在 hard knowledge 约束下将非空相对怀疑集合确定性归一化为玩家 belief row。
- `werewolf/models/twd_tom/annotation_v2.py`：严格校验并绑定既有 Speech/Belief V2 sidecar，不回写 canonical 数据。
- `werewolf/models/twd_tom/dataset.py`：唯一 tom-v2 Dataset；显式区分历史 `legacy_v1`（empty→admissible uniform）与 `v1_empty_unobserved`，输出 7×7 target、scope/observed/supervision masks 与 diagonal target mask。
- `werewolf/models/twd_tom/dense_dataset.py`：将同一局的所有 strict-PRE snapshot 组织为共享公开序列与多监督边界；超过 `max_seq_len` 后无法保持精确前缀的数据直接拒绝。
- `werewolf/models/twd_tom/baselines.py`：只用 fold 训练标签拟合 observer global prior 与 observer+phase prior，不读取 fold validation 或 test label。
- `script/twd_tom/materialize_canonical_belief_dataset.py`：验证成功批次摘要链，将 canonical per-game snapshots 确定性分配为 game-level train/validation/test JSONL，并写入 split manifest。
- `script/twd_tom/audit_canonical_belief_data.py`：在物化前验证成功批次摘要链和 canonical label 可训练性，并报告 support/sequence/truncation 统计。
- `script/twd_tom/audit_dense_belief_dataset.py`：训练前证明每条监督对应 exact strict-PRE encoded prefix，并报告逐局边界与序列长度。
- `script/twd_tom/materialize_development_folds.py`：只合并原 train+validation，生成开发集 5-fold；test 只保留 manifest 身份，不复制或读取其物理数据。
- `script/twd_tom/run_development_oof.py`：按 8 局一个 batch 运行 dense 5-fold OOF、早停、逐局指标、game bootstrap CI 与训练集 prior 基线。
- `script/twd_tom/audit_belief_label_repeatability.py`：审计 3–5 份相同 frozen-state V2 replicate 的 exact/Jaccard/TV/JS。
- `script/twd_tom/run_annotation_v2_ablation.py`：运行 Speech V1/V2 × Belief `v1_empty_unobserved`/V2 的固定五折归因矩阵；exploratory 运行不依赖 repeatability，冻结正式 benchmark 时才强制要求通过审计。
- `script/twd_tom/export_belief_worst_cases.py`：从每折最佳 checkpoint 导出 `legacy_v1`、`v1_empty_unobserved`、V2 target 对照与 boundary 邻域误差样本。
- `script/twd_tom/audit_shadow_speech_parser.py`：只读重解析既有公开发言，比较 DeepSeek 与原 parser 的状态、动作顺序和动作集合；结果写入独立目录，不能替换 canonical annotation。
- `script/twd_tom/collection_budget.py`：执行正式采集的 gameplay、belief、total call 和单局墙钟预算。
- `script/twd_tom/replay_canonical_trajectory.py`：不调用模型地重放 canonical submitted actions，并核对逐步状态、公开事件和 observer views。
- `werewolf/models/twd_tom/belief_backbone.py`：公开历史与 observer query 编码；直接输出 7×7 belief logits。
- `werewolf/models/twd_tom/losses.py`：存活 observer、非对角 target 上的 masked belief distribution loss。
- `werewolf/models/twd_tom/metrics.py`：soft-target support-hit 指标与 uniform non-self baseline。
- `werewolf/models/twd_tom/checkpoint.py`：训练、评估和 gameplay 共用的严格 checkpoint 恢复契约。
- `werewolf/models/twd_tom/inference.py`：按完整 strict-PRE 公开历史重算 `belief_logits[7,7]` 与行归一化 `belief_matrix[7,7]`，不维护 recurrent belief state。
- `script/twd_tom/train.py` 与 `eval.py`：单一 tom-v2 objective 的训练、checkpoint 与评估入口。
- `tests/twd_tom/`：采集、时间边界和只读性回归测试。

当前冻结版本为 `classic7_public_event_sequence_v4`、`classic7_speech_annotation_v3`、`classic7_speech_action_v1`、`classic7_public_speech_realization_prompt_v1`、`classic7_pre_speech_player_suspicion_v5` 和 `classic7_pre_speech_player_suspicion_prompt_v6`。完整字段、ontology 与失败策略见 `docs/public_speech_event_contract.md`。

训练 Dataset 默认启用通用座位循环旋转；验证与 evaluation 保持原始座位。旋转同时覆盖 observer、target、公开事件玩家引用、belief matrix 和 masks。dense 模式的 `batch_size` 表示局数而不是 snapshot 数；同一局所有合法 PRE 边界共同产生监督。

## 安装与测试

```bash
python -m pip install -e .
python -m pytest -q
python -m compileall -q werewolf script tests
```
