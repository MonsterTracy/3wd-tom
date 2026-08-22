# 仓库结构

## tom-v2 当前 active path

| 路径 | 职责 |
|---|---|
| `run_random.py` | 在公开发言前触发唯一 self-report collector |
| `werewolf/speech/private_belief_perceiver.py` | 构造并解析 playing-agent private readonly query |
| `werewolf/models/twd_tom/belief_snapshot.py` | 取得 observer 合法信息并检查 agent 状态不变 |
| `werewolf/models/twd_tom/samples.py` | 冻结 public cutoff，保存符号 belief snapshot |
| `werewolf/models/twd_tom/collector.py` | 将 raw sample 写入 JSONL |
| `werewolf/models/twd_tom/belief_labels.py` | 将相对怀疑集合确定性转换为归一化 belief row |
| `werewolf/models/twd_tom/dataset.py` | 输出 7×7 observer-conditioned target 与两个 mask |
| `tests/twd_tom/` | 时间边界、只读性、符号标签和写入契约测试 |

## 本阶段保留但不重构的计算层

`werewolf/models/twd_tom/belief_backbone.py`、loss、metrics、`script/twd_tom/train.py` 与 `eval.py` 保持不变，等待后续阶段与新 Dataset target 对齐。

## 已退出 tom-v2 主线

仓库不再提供 public-only reporter、PBM、external offline annotation/materialization、D splitter、online ToM2 shadow 或 tom-v1 archive 的可导入实现。tom-v1 历史由 Git 保存。
