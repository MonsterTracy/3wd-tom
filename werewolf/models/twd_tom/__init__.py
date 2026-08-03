"""Actor-perspective ToM data and frozen model components for classic seven."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "PublicEventFeatureBuilder": "action_features",
    "ToMBeliefBackbone": "belief_backbone",
    "ToMBeliefBackboneConfig": "belief_backbone",
    "suspicion_set_to_pair_target": "belief_labels",
    "pair_probabilities_to_belief_marginals": "belief_labels",
    "PlayingAgentBeliefSnapshotCollector": "belief_snapshot",
    "TWDToMSampleCollector": "collector",
    "TWDToMDataset": "dataset",
    "collate_twd_tom_samples": "dataset",
    "load_twd_tom_jsonl": "dataset",
    "masked_distribution_cross_entropy": "losses",
    "masked_distribution_kl_divergence": "losses",
    "compute_subjective_pair_diagnostics": "metrics",
    "compute_subjective_pair_metrics": "metrics",
    "SAMPLE_SCHEMA_VERSION": "samples",
    "make_twd_tom_sample": "samples",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
