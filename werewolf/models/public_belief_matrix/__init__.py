"""Public Belief Matrix V1 components.

    structured public history
        -> observer x target normalized suspicion matrix

The package currently provides deterministic target contracts, a shared
structured public-prefix contract, a causal matrix backbone, masked row loss,
deterministic matrix metrics, a stateless reporter, raw symbolic collection,
and a strict materialized Dataset contract. Training remains an independent CLI.
"""

from werewolf.models.public_belief_matrix.public_prefix import (
    build_public_belief_matrix_visible_prefix,
    render_public_belief_matrix_visible_prefix,
)
from werewolf.models.public_belief_matrix.backbone import (
    PublicBeliefMatrixBackbone,
    PublicBeliefMatrixBackboneConfig,
)
from werewolf.models.public_belief_matrix.losses import (
    masked_row_soft_target_cross_entropy,
)
from werewolf.models.public_belief_matrix.metrics import (
    masked_mean_row_cross_entropy,
    masked_mean_row_entropy,
    mean_observer_pairwise_tv,
    mean_prediction_diagonal_mass,
)
from werewolf.models.public_belief_matrix.targets import (
    PublicBeliefMatrixTarget,
    suspicion_reports_to_matrix_target,
    suspicion_set_to_row_target,
)

__all__ = [
    "PublicBeliefMatrixBackbone",
    "PublicBeliefMatrixBackboneConfig",
    "PublicBeliefMatrixTarget",
    "build_public_belief_matrix_visible_prefix",
    "render_public_belief_matrix_visible_prefix",
    "masked_mean_row_cross_entropy",
    "masked_mean_row_entropy",
    "masked_row_soft_target_cross_entropy",
    "mean_observer_pairwise_tv",
    "mean_prediction_diagonal_mass",
    "suspicion_reports_to_matrix_target",
    "suspicion_set_to_row_target",
]
