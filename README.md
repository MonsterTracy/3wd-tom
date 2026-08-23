# 3WD-ToM

3WD-ToM 是一个面向经典七人狼人杀的 agent-centric Theory of Mind 研究项目。

当前游戏规则基线固定为 2 狼人、1 预言家、1 女巫、3 村民：无存活狼人时好人胜；存活狼人数达到或超过其他存活玩家总数时狼人胜。

当前 tom-v2 研究问题是：给定时间点 `t` 之前的公开历史和目标观察者 `i`，预测该观察者当前对其他玩家的狼人怀疑 belief：

```text
f(public_history <= t, observer_id=i) -> B_t(i, j)
```

## 当前唯一标签来源

训练标签只来自 playing agent 的 readonly self-report。系统在公开发言前冻结合法时间边界，对每个存活玩家调用其私有 belief query：

```text
你当前怀疑哪些玩家是狼人？
```

原始标签保存为 `suspected_werewolves` 符号集合。该集合必须包含观察者已知的其他狼人，且不得包含观察者自身或已知非狼人；违反约束的报告记为 `semantic_error`，不修复也不进入训练。采集阶段不生成概率分布、belief matrix、pair target 或 21 类投影。

## 当前数据流

```text
Classic-7 simulator
  -> 发言前冻结 public history
  -> playing-agent readonly private belief query
  -> 同一份 speaker PRE belief 只读交给紧随其后的 day cognition
  -> 冻结公开表达意图并由第二次 LLM 调用生成自然中文
  -> 独立 speech parser 只从公开原文生成版本化 annotation sidecar
  -> raw belief snapshot JSONL
  -> canonical game-level train/validation/test materialization
  -> deterministic sparse suspicion-set conversion
  -> 7×7 observer-conditioned belief Dataset
  -> observer-query belief backbone
  -> masked belief distribution training/evaluation
```

public-only reporter、external offline annotation、PBM、ToM1/ToM2 双任务和旧 formal reporter 不属于 tom-v2 主线。

公开事件与发言标注采用分层契约：`public_events` 的 `public_speech` 只保存 `event_idx`、`speaker` 和自然语言 `raw_text`；人工定义的 14 类语义动作、解析器版本、原始解析响应和错误状态单独保存到 `speech_annotations.jsonl`。两者通过 `event_idx`、speaker 和原文 SHA256 完整绑定。canonical 数据要求每条公开发言都由独立 parser 成功标注；不允许把生成器的冻结 intent 直接当成训练输入。当前 belief backbone 只编码结构化动作，不编码 `raw_text`，但原文永久保留，未来更换 ontology/parser 时可以重新标注而无需重跑游戏。

## 代码位置

- `run_random.py`：游戏循环与唯一 belief collector 接入点。
- `werewolf/speech/private_belief_perceiver.py`：playing-agent readonly self-report。
- `werewolf/models/twd_tom/belief_snapshot.py`：逐观察者只读状态检查。
- `werewolf/models/twd_tom/samples.py`：冻结公开时间边界并生成 raw sample。
- `werewolf/models/twd_tom/public_events.py`：v4 无损公开事件序列与 raw-text-free 模型投影。
- `werewolf/models/twd_tom/speech_annotations.py`：版本化发言语义 sidecar、原文 digest 和完整绑定校验。
- `werewolf/speech/speech_perceiver.py`：独立 14-action 公开发言解析器。
- `werewolf/models/twd_tom/collector.py`：写入 raw JSONL。
- `werewolf/models/twd_tom/belief_labels.py`：在 hard knowledge 约束下将相对怀疑集合确定性归一化为玩家 belief row；空集合只在 admissible players 上均分。
- `werewolf/models/twd_tom/dataset.py`：唯一 tom-v2 Dataset；输出 7×7 target、observer alive mask 与 diagonal target mask。
- `script/twd_tom/materialize_canonical_belief_dataset.py`：将 canonical per-game snapshots 确定性分配为 game-level train/validation/test JSONL。
- `script/twd_tom/audit_canonical_belief_data.py`：在物化前验证 canonical label 可训练性，并报告 support/sequence/truncation 统计。
- `script/twd_tom/collection_budget.py`：执行正式采集的 gameplay、belief、total call 和单局墙钟预算。
- `script/twd_tom/replay_canonical_trajectory.py`：不调用模型地重放 canonical submitted actions，并核对逐步状态、公开事件和 observer views。
- `werewolf/models/twd_tom/belief_backbone.py`：公开历史与 observer query 编码；直接输出 7×7 belief logits。
- `werewolf/models/twd_tom/losses.py`：存活 observer、非对角 target 上的 masked belief distribution loss。
- `werewolf/models/twd_tom/metrics.py`：soft-target support-hit 指标与 uniform non-self baseline。
- `script/twd_tom/train.py` 与 `eval.py`：单一 tom-v2 objective 的训练、checkpoint 与评估入口。
- `tests/twd_tom/`：采集、时间边界和只读性回归测试。

当前冻结版本为 `classic7_public_event_sequence_v4`、`classic7_speech_annotation_v1`、`classic7_speech_action_v1` 和 `classic7_pre_speech_player_suspicion_v4`。完整字段、ontology 与失败策略见 `docs/public_speech_event_contract.md`。

训练 Dataset 默认启用通用座位循环旋转；验证与 evaluation 保持原始座位。旋转同时覆盖 observer、target、公开事件玩家引用、belief matrix 和 masks。

## 安装与测试

```bash
python -m pip install -e .
python -m pytest -q
python -m compileall -q werewolf script tests
```
