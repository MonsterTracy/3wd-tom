"""Observer-specific second-order ToM components for seven-player Werewolf.

The package contains a raw collection path and a separate pair-model path:

    complete structured public-event prefixes
        -> playing-agent player-level suspicion sets

    explicitly projected pair samples
        -> causal belief backbone
        -> masked subjective-belief loss and metrics

Truth-derived role labels and the legacy ten-field event representation
are intentionally excluded.
"""

from werewolf.models.twd_tom.action_features import (
    PublicEventFeatureBuilder,
)
from werewolf.models.twd_tom.belief_backbone import (
    ToMBeliefBackbone,
    ToMBeliefBackboneConfig,
)
from werewolf.models.twd_tom.belief_labels import (
    pair_probabilities_to_belief_marginals,
    suspicion_set_to_pair_target,
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
    masked_distribution_cross_entropy,
    masked_distribution_kl_divergence,
)
from werewolf.models.twd_tom.metrics import (
    compute_subjective_pair_metrics,
)
from werewolf.models.twd_tom.samples import (
    SAMPLE_SCHEMA_VERSION,
    make_twd_tom_sample,
)


__all__ = [
    "PublicEventFeatureBuilder",
    "ToMBeliefBackbone",
    "ToMBeliefBackboneConfig",
    "suspicion_set_to_pair_target",
    "pair_probabilities_to_belief_marginals",
    "PlayingAgentBeliefSnapshotCollector",
    "TWDToMSampleCollector",
    "TWDToMDataset",
    "collate_twd_tom_samples",
    "load_twd_tom_jsonl",
    "masked_distribution_cross_entropy",
    "masked_distribution_kl_divergence",
    "compute_subjective_pair_metrics",
    "SAMPLE_SCHEMA_VERSION",
    "make_twd_tom_sample",
]
