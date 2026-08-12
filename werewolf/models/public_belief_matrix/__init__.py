"""Public Belief Matrix V1 deterministic target and input contracts.

    structured public history
        -> observer x target normalized suspicion matrix

This package does not implement collection, a model, or training.
"""

from werewolf.models.public_belief_matrix.public_prefix import (
    build_public_belief_matrix_visible_prefix,
)
from werewolf.models.public_belief_matrix.targets import (
    PublicBeliefMatrixTarget,
    suspicion_reports_to_matrix_target,
    suspicion_set_to_row_target,
)

__all__ = [
    "PublicBeliefMatrixTarget",
    "build_public_belief_matrix_visible_prefix",
    "suspicion_reports_to_matrix_target",
    "suspicion_set_to_row_target",
]
