# 仓库结构

## tom-v2 当前 active path

| 路径 | 职责 |
|---|---|
| `run_random.py` | 在公开发言前触发唯一 self-report collector，并把 speaker 同一份冻结报告交给 day cognition |
| `werewolf/speech/private_belief_perceiver.py` | 构造并解析 playing-agent private readonly query |
| `werewolf/models/twd_tom/belief_snapshot.py` | 取得 observer 合法信息并检查 agent 状态不变 |
| `werewolf/models/twd_tom/samples.py` | 冻结 public cutoff，保存符号 belief snapshot |
| `werewolf/models/twd_tom/collector.py` | 将 raw sample 写入 JSONL |
| `script/twd_tom/collection_budget.py` | 对正式采集的实际 backend dispatch 和单局墙钟执行预算 |
| `script/twd_tom/audit_canonical_belief_data.py` | 物化前的 canonical label/sequence/truncation 审计 |
| `script/twd_tom/audit_shadow_speech_parser.py` | 对既有公开发言执行只读 DeepSeek 影子解析并输出跨模型一致性审计，不回写 canonical 标注 |
| `script/twd_tom/materialize_canonical_belief_dataset.py` | 按 game-level split 发布不变的 raw snapshots |
| `werewolf/models/twd_tom/belief_labels.py` | 校验 hard knowledge 并将相对怀疑集合确定性转换为归一化 belief row |
| `werewolf/models/twd_tom/dataset.py` | 输出 7×7 observer-conditioned target 与两个 mask |
| `werewolf/models/twd_tom/dense_dataset.py` | 以 game 为 batch unit，验证并组织全部 strict-PRE boundary |
| `werewolf/models/twd_tom/baselines.py` | fold-train-only global/phase empirical prior |
| `script/twd_tom/audit_dense_belief_dataset.py` | 证明多边界监督的 exact-prefix 因果契约 |
| `script/twd_tom/materialize_development_folds.py` | 从 train+validation 构造 development-only 5-fold，封存 test |
| `script/twd_tom/run_development_oof.py` | dense 5-fold 训练、逐局 OOF 聚合与 bootstrap CI |
| `tests/twd_tom/` | 时间边界、只读性、符号标签和写入契约测试 |

## 计算层

`werewolf/models/twd_tom/belief_backbone.py`、loss、metrics、`script/twd_tom/train.py` 与 `eval.py` 已与唯一的 7×7 Dataset target 对齐。训练可在一次 game forward 中读取多个 PRE boundary；原单 boundary inference 不变。metrics 使用 soft-target top-1 support hit，并同步报告 uniform non-self、fold-train global prior 与 phase prior。

## 已退出 tom-v2 主线

仓库不再提供 public-only reporter、PBM、可进入训练链的 external offline annotation/materialization、D splitter、online ToM2 shadow 或 tom-v1 archive 的可导入实现。DeepSeek speech-parser shadow 只生成独立审计工件，不是标签替换或训练数据来源。tom-v1 历史由 Git 保存。
