# tom-v2 belief snapshot 契约

## 时间边界

每个 raw sample 在 `speech` 或 `speech_pk` 动作生成前创建。冻结的 `public_events` 必须截止于与当前 speaker 对应的 `turn_start`，且 `label_cutoff_step_idx == step_idx`。

## 观察者

采集对象是当前公开存活玩家。对每个 `observer_id`：

1. Environment 提供该玩家在当前边界合法可见的 observation；
2. Environment 提供该玩家的确定性 hard knowledge；
3. 调用该 playing agent 的 `report_suspected_werewolves_readonly()`；
4. query 前后 agent-owned state 必须完全相同。

不得向 collector 注入全局真实角色、其他玩家私有信息或未来事件。

## 原始标签

成功报告的原始标签是按 canonical player 顺序保存的 `suspected_werewolves` 集合。失败报告保存明确 status/error，并将该观察者的 suspicion 记为 `null`；不进行修复、重试、fallback、补猜或概率化。

raw sample 保存 public history 的两个 digest、观察者集合、每个观察者的 self-report、hard knowledge、状态与 backend provenance。它不保存 raw model response、私有 observation、pair target 或概率矩阵。

## Dataset 契约

tom-v2 只有一个 Dataset。输入是同一时间边界的结构化公开历史和存活观察者集合；raw label 直接读取 playing-agent self-report 中的 `suspected_werewolves`，不经过 external annotation、public-only reporter、ToM1/ToM2 materialization 或旧 lineage adapter。

Dataset 将每个观察者的怀疑集合确定性转换为长度 7 的稀疏 belief row：非空集合只在被列入的玩家上均分概率，其他玩家为 0；空集合对应六个非自身玩家的均匀分布。观察者自身不能出现在怀疑集合中且对角线恒为 0。完整 target 为固定 `7×7`，行是 observer，列是 target player。

Dataset 输出：

- `belief_targets`：`[7, 7]` 浮点张量；
- `observer_alive_mask`：`[7]` 布尔张量，只表示该时间点公开存活的 observer；
- `diagonal_target_mask`：`[7, 7]` 布尔张量，只排除 observer 自身列。

死亡玩家仍是合法 target，因此不得根据存活状态屏蔽 target 列。raw sample 中的 hard knowledge 只保留为 provenance，并用于合法性审计与泄漏检查；它不约束、补充或删除 self-report 内容，也不进入模型特征或 Dataset 输出。训练 Dataset 对非 `ok` 的存活 observer 直接失败，不补猜、不生成额外有效性 mask。

## Canonical materialization

`script/twd_tom/materialize_canonical_belief_dataset.py` 扫描 `canonical_root/games/*/belief_snapshots.jsonl`，按稳定 SHA256 排名将完整 game 分配到 train、validation、test。一个 game 的所有 PRE snapshots 必须进入同一 split；输出记录保留原始 provenance 和 PRE boundary metadata，不执行 snapshot-level split。

## 唯一来源

`label_source = playing_agent_readonly_self_report`。public-only reporter、external offline annotation 和 PBM 不是合法训练标签来源。

## 模型与目标函数

模型输入只有结构化 `public_history <= t`，并对七个 canonical observer query 共享参数。输出 `belief_logits[B, 7, 7]`，直接对应 Dataset 的 `belief_targets[B, 7, 7]`。

对每个存活 observer，在六个非自身 target 上计算 soft-target cross entropy；batch loss 是所有存活 observer 行的算术平均。死亡 observer 行不参与监督，死亡 player 列仍参与其他 observer 的分布。checkpoint、metrics、train 和 eval 均使用这一单一目标，不保存旧任务阶数或组合类别字段。
