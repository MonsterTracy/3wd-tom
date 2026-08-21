# Legacy-to-mainline audit

Audit baseline: `e2eadb12e621edce633c9044c2f9e0f7b58d4ac0`

This source-ownership audit was implemented as a non-destructive migration to
`archive/legacy_tom`. It did not change gameplay, data schemas, or runtime
behavior. In this document, **archive** means retained for historical
reproducibility outside current-mainline documentation and normal entry
points; it does not mean deleted.

The original paths in the inventory tables identify the audited locations.
Their source now lives under the corresponding `archive/legacy_tom/` prefix.
Canonical hard-knowledge ownership is `werewolf/observer_knowledge.py`.

## Current research mainline

The current mainline is:

```text
Simulator
  run_random.eval(..., trajectory_recorder=...)
    -> A: classic7_game_interaction_trajectory_v1
    -> C0: classic7_observer_view_provenance_v1

A + C0
  werewolf.offline_annotation
    -> C1: classic7_offline_annotation_record_v1

A + C0 + C1
  werewolf.offline_materialization
    -> D ToM1: offline_private_conditioned_tom1_v1
    -> D ToM2: offline_public_only_tom2_v1

D ToM1 + D ToM2
  script.twd_tom.split_offline_d_training_data
    -> game-disjoint train / validation / test JSONL

split D records
  werewolf.models.twd_tom.dataset.TWDToMDataset
    -> script.twd_tom.train
    -> script.twd_tom.eval
```

The mainline ownership points are therefore:

| Stage | Keep as canonical owner |
|---|---|
| Simulator integration | `run_random.py` and the Environment/agent stack |
| A/C0 capture | `werewolf/trajectory.py`, `script/twd_tom/collect_canonical_trajectories.py` |
| C1 annotation | `werewolf/offline_annotation.py` |
| D materialization | `werewolf/offline_materialization.py` |
| D game split | `script/twd_tom/split_offline_d_training_data.py` |
| Dataset/features | `werewolf/models/twd_tom/dataset.py` and its `twd_tom` dependencies |
| Training/evaluation | `script/twd_tom/train.py`, `script/twd_tom/eval.py` |

C1 and D retain their library APIs:
`annotate_pre_speech_suspicion()` / `write_annotation_jsonl()` and
`materialize_offline_tom_records()` / `write_offline_tom_jsonl()`. The
filesystem orchestration gap is closed by
`script/twd_tom/materialize_canonical_dataset.py`, which composes those APIs
without reintroducing online label collection.

## Legacy inventory: `script/tom`

The whole package is the earlier formal-ToM pilot and is not part of the D
lineage.

| File | What it owns | Recommendation |
|---|---|---|
| `script/tom/__init__.py` | Namespace description for the formal pilot | **Archive** with the pilot |
| `script/tom/split_pilot.py` | `random.Random` whole-game split of legacy raw formal rows; emits `train/val/test` and the old manifest shape | **Archive**; do not merge with the canonical D hash splitter |
| `script/tom/train.py` | Training/evaluation of `werewolf.models.tom.BeliefModel` on `TomDataset` and 7-way observer targets | **Archive**; do not merge its model or target contract into TWD ToM training |

Why this is not a second current training implementation: its records,
manifest shape, event encoding, target tensor, and model are all different
from canonical D plus `TWDToMDataset`. Merging the implementations would blur
lineage rather than remove duplication.

## Legacy inventory: `werewolf/models/tom`

This package is an internally coherent earlier formal-ToM stack. Most of it
has only legacy runtime/CLI and `tests/tom` consumers. Its one former C1
dependency was separated during the archive migration.

| File | What it owns | Recommendation |
|---|---|---|
| `werewolf/models/tom/__init__.py` | Re-exports the complete legacy formal stack | **Archived** |
| `werewolf/models/tom/collection.py` | `Collector`: post-committed-speech online observer reporting and raw JSONL writes | **Archive** |
| `werewolf/models/tom/dataset.py` | Legacy `TomDataset`, flat formal-event encoding, and 7x7 target materialization | **Archive** |
| `werewolf/models/tom/losses.py` | Loss for the legacy 7-player target rows | **Archive** |
| `werewolf/models/tom/model.py` | Legacy `BeliefModel` encoder and 7-player output head | **Archive** |
| `werewolf/models/tom/public_history.py` | Projection from public-event ledger into the legacy flat event vocabulary | **Archive** |
| `werewolf/models/tom/reporter.py` | Online reporter plus pure legal-state/hard-knowledge derivation | **Archived; delegates pure helpers to canonical ownership** |
| `werewolf/models/tom/schema.py` | Legacy player/action/event/phase IDs and targeted-only `SpeechAction` | **Archived** |
| `werewolf/models/tom/targets.py` | Subjective suspicion set to legacy 7-way rows / 7x7 target | **Archive** |

The pre-migration dependency was:

```text
werewolf/offline_annotation.py
  -> BeliefReporter.legal_state()
  -> BeliefReporter.derive_hard_knowledge()
  -> werewolf.models.tom.schema.PLAYER_NAMES / normalize_player()
```

The migration moved only those deterministic observation-selection and hard-
knowledge operations to `werewolf/observer_knowledge.py`. Canonical offline
annotation imports that module directly. The archived `BeliefReporter`
delegates to it while retaining its backend defaults, prompt, and legacy
collection record contract.

The rest of this package should not be merged into
`werewolf/models/twd_tom`:

- the legacy target is a 7-player distribution for each observer, while the
  current TWD target is a 21-class wolf-pair distribution;
- the legacy model and categorical event vocabulary are not the current
  `PublicEventFeatureBuilder` / `ToMBeliefBackbone` contract;
- legacy `SpeechAction` requires a player object and cannot represent the
  canonical targetless `abstain_intent` and `no_commitment` events whose raw
  object is JSON `null`;
- preserving the old implementation as a labeled historical pilot is safer
  than creating adapters between incompatible persisted lineages.

## Duplicate collection pipelines

Three ToM-oriented collection paths coexist around gameplay.

| Path | Capture boundary and output | Downstream | Disposition |
|---|---|---|---|
| Canonical A/C0 recorder | `CanonicalGameInteractionTrajectoryRecorder`; records every committed transition plus PRE/POST observer-view provenance | Offline C1, D, D splitter, `TWDToMDataset` | **Keep; sole current-mainline collection path** |
| Legacy online TWD snapshot collector | `run_random.build_twd_tom_sample_collector()` -> `TWDToMSampleCollector`; calls belief reporters during gameplay around public speech and writes V2.7 raw samples | `materialize_training_data.py`, projection and legacy split tools | **Archive as one legacy lineage** |
| Legacy formal post-speech collector | `run_random.build_tom_collector()` -> `werewolf.models.tom.collection.Collector`; calls `BeliefReporter` after committed speech and writes formal pilot rows | `script/tom/split_pilot.py`, `script/tom/train.py` | **Archive as one legacy pilot** |

`run_random.eval()` currently accepts all three hooks. It calls the canonical
recorder before agent action, after agent action, and after `env.step`; it can
also invoke the online TWD collector before/after speech and the formal
collector after speech. That is the central coexistence point. Retain the
simulator behavior, but after legacy consumers are archived, remove or isolate
only the legacy collector wiring in a separate cleanup. Do not change action,
phase, observation, or Environment semantics as part of that cleanup.

There is also a stale `run_battle.py` collection path. It constructs
`TWDToMSampleCollector` without the now-required `snapshot_collector` and calls
`record()` with the older `(observation, roles)` shape. It is both a duplicate
entry point and evidence of API drift; it should not be treated as a current
collection route.

The following are adjacent but are not duplicate ToM training-data collectors:

- `tom2_shadow` performs online checkpoint shadow inference/logging;
- the public-belief-matrix collector owns a separate PBM experiment;
- call-audit logging records backend request provenance rather than ToM
  training labels.

## Duplicate legacy entry points under `script/twd_tom`

Several scripts are wrappers around the same online V2.7 raw-snapshot path,
not independent canonical pipelines:

| Files | Recommendation |
|---|---|
| `collect.py`, `real_backend_dry_run.py`, `monitored_collection.py`, `formal_batch_collection.py` | **Archive together** as online-collection utilities after any required historical runbook is captured |
| `pipeline.py` | **Merge/repurpose** only its explicit-stage CLI shell if a unified canonical CLI is desired; its current collect/project/split stages route through the old online lineage and must not remain documented as canonical |
| `materialize_training_data.py`, `build_dev100_training_data.py` | **Archive together**; these materialize the pre-D V2.7 lineage |
| `project_suspicion_to_pairs.py`, `split_formal_dataset.py`, `split_training_data.py` | **Archive together**; canonical D validation/materialization and `split_offline_d_training_data.py` supersede this path |
| `collect_canonical_trajectories.py`, `split_offline_d_training_data.py`, `train.py`, `eval.py` | **Keep** as current-mainline entry points |
| `audit_training_targets.py` | **Keep** as a read-only Dataset audit, provided it continues to accept strictly validated D rows |

Before migration, `script/twd_tom/pipeline.py`, `README.md`,
`docs/architecture.md`, and `docs/repository_structure.md` called online
raw-snapshot collection canonical. The pipeline now resides in the archive,
and the current docs identify A/C0 -> C1 -> D as the mainline.

## Non-destructive sequence and remaining cleanup

1. **Completed — declare the mainline.** Top-level documentation now points to
   A/C0 -> C1 -> D -> D split -> train/eval. Legacy commands are explicitly
   historical.
2. **Completed — close the old-package dependency.** C1's exact legal-state
   and hard-knowledge behavior now belongs to canonical annotation support,
   with equivalence coverage.
3. **Completed — archive the formal pilot as a unit.** `script/tom` and
   `werewolf/models/tom` moved together; their target/model/schema were not
   merged into TWD ToM.
4. **Completed — archive the online V2.7 lineage as a unit.** Its collection
   wrappers, raw materializer, projection, split utilities, and tests use the
   archive imports. Historical artifacts and commands remain reproducible.
5. **Deferred compatibility cleanup.** `run_random.py` retains the explicit
   legacy formal-collector hook by importing the archive. Removing that hook
   or the stale `run_battle.py` option would alter an existing opt-in runtime
   surface and is intentionally outside this no-semantics-change migration.
6. **Not performed — CLI replacement.** The old `pipeline.py` is archived;
   no new automatic orchestrator or compatibility adapter was introduced.

## Keep/archive/merge summary

- **Keep:** canonical trajectory, offline annotation/materialization, D hash
  split, TWD Dataset/features/model/loss, and current TWD train/eval.
- **Archived:** the formal pilot model/training stack and online V2.7 sample
  collection/processing lineage, preserving history and reproducibility.
- **Merged:** only the pure hard-knowledge helpers required by C1. Legacy
  schemas, targets, models, collectors, and persisted records remain separate.

No recommendation above requires a gameplay-semantic change. The proposed
boundary is about which artifact lineage owns collection and training, not how
agents choose actions or how the Environment resolves them.
