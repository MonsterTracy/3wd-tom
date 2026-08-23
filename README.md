# 3WD-ToM

3WD-ToM 是一个面向经典七人狼人杀的 agent-centric Theory of Mind 研究项目。

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
  -> raw belief snapshot JSONL
  -> canonical game-level train/validation/test materialization
  -> deterministic sparse suspicion-set conversion
  -> 7×7 observer-conditioned belief Dataset
  -> observer-query belief backbone
  -> masked belief distribution training/evaluation
```

public-only reporter、external offline annotation、PBM、ToM1/ToM2 双任务和旧 formal reporter 不属于 tom-v2 主线。

## 代码位置

- `run_random.py`：游戏循环与唯一 belief collector 接入点。
- `werewolf/speech/private_belief_perceiver.py`：playing-agent readonly self-report。
- `werewolf/models/twd_tom/belief_snapshot.py`：逐观察者只读状态检查。
- `werewolf/models/twd_tom/samples.py`：冻结公开时间边界并生成 raw sample。
- `werewolf/models/twd_tom/collector.py`：写入 raw JSONL。
- `werewolf/models/twd_tom/belief_labels.py`：在 hard knowledge 约束下将相对怀疑集合确定性归一化为玩家 belief row；空集合只在 admissible players 上均分。
- `werewolf/models/twd_tom/dataset.py`：唯一 tom-v2 Dataset；输出 7×7 target、observer alive mask 与 diagonal target mask。
- `script/twd_tom/materialize_canonical_belief_dataset.py`：将 canonical per-game snapshots 确定性分配为 game-level train/validation/test JSONL。
- `script/twd_tom/audit_canonical_belief_data.py`：在物化前验证 canonical label 可训练性，并报告 support/sequence/truncation 统计。
- `script/twd_tom/collection_budget.py`：执行正式采集的 gameplay、belief、total call 和单局墙钟预算。
- `werewolf/models/twd_tom/belief_backbone.py`：公开历史与 observer query 编码；直接输出 7×7 belief logits。
- `werewolf/models/twd_tom/losses.py`：存活 observer、非对角 target 上的 masked belief distribution loss。
- `werewolf/models/twd_tom/metrics.py`：soft-target support-hit 指标与 uniform non-self baseline。
- `script/twd_tom/train.py` 与 `eval.py`：单一 tom-v2 objective 的训练、checkpoint 与评估入口。
- `tests/twd_tom/`：采集、时间边界和只读性回归测试。

训练 Dataset 默认启用通用座位循环旋转；验证与 evaluation 保持原始座位。旋转同时覆盖 observer、target、公开事件玩家引用、belief matrix 和 masks。

## 安装与测试

```bash
python -m pip install -e .
python -m pytest -q
python -m compileall -q werewolf script tests
```
