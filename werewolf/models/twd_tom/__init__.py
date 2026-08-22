"""七人狼人杀的 playing-agent belief self-report 与 tom-v2 Dataset。

当前唯一原始标签路径在公开发言前冻结时间边界，调用目标 playing agent 的
readonly private belief query，并保存 ``suspected_werewolves`` 符号集合。Dataset
将该集合确定性转换为固定 7×7 observer-conditioned belief target。
"""

from werewolf.models.twd_tom.action_features import (
    PublicEventFeatureBuilder,
)
from werewolf.models.twd_tom.belief_backbone import (
    ToMBeliefBackbone,
    ToMBeliefBackboneConfig,
)
from werewolf.models.twd_tom.belief_labels import (
    suspicion_set_to_belief_vector,
)
from werewolf.models.twd_tom.belief_snapshot import (
    PlayingAgentBeliefSnapshotCollector,
)
from werewolf.models.twd_tom.collector import (
    TWDToMSampleCollector,
)
from werewolf.models.twd_tom.dataset import (
    TWDToMDataset,
    collate_twd_tom_samples,
    load_twd_tom_jsonl,
)
from werewolf.models.twd_tom.losses import (
    masked_belief_distribution_loss,
    masked_belief_probabilities,
)
from werewolf.models.twd_tom.metrics import (
    compute_belief_metrics,
)
from werewolf.models.twd_tom.samples import (
    SAMPLE_SCHEMA_VERSION,
    make_twd_tom_sample,
)


__all__ = [
    "PublicEventFeatureBuilder",
    "ToMBeliefBackbone",
    "ToMBeliefBackboneConfig",
    "suspicion_set_to_belief_vector",
    "PlayingAgentBeliefSnapshotCollector",
    "TWDToMSampleCollector",
    "TWDToMDataset",
    "collate_twd_tom_samples",
    "load_twd_tom_jsonl",
    "masked_belief_distribution_loss",
    "masked_belief_probabilities",
    "compute_belief_metrics",
    "SAMPLE_SCHEMA_VERSION",
    "make_twd_tom_sample",
]
