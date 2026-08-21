# Repository structure and storage boundary

## Source tree

| Path | Responsibility |
|---|---|
| `werewolf/agents/` | Agent abstractions, Planner/Renderer orchestration, prompt and action contracts |
| `werewolf/backends/` | Named OpenAI-compatible backend loading and request handling |
| `werewolf/envs/` | Seven-player game state, observations, valid actions, phases, and public events |
| `werewolf/speech/` | Online/offline speech parsing and playing-agent belief reports |
| `werewolf/models/twd_tom/` | ToM schemas, collection helpers, Dataset, backbones, targets, losses, metrics, and shadow inference |
| `script/twd_tom/` | Canonical A/C0 collection, D split, audit, training, and evaluation entry points |
| `archive/legacy_tom/` | Importable historical formal-ToM and online V2.7 collection/processing code |
| `tests/` | Self-contained deterministic tests organized by subsystem |
| `configs/` | Reusable runtime profiles; see `configs/README.md` |
| `docs/` | Research contracts, architecture, provenance, and deployment guidance |

The root entry points `run_battle.py`, `run_random.py`, and `run_batch.sh`
remain part of the gameplay interface. The repository does not contain a
`jobs/` directory or an in-repository model-service manager.

## Collection entry points

`python -m script.twd_tom.collect_canonical_trajectories` is the canonical
gameplay collection interface. It calls `run_random.eval()` with only the
canonical trajectory recorder and emits paired A/C0 artifacts.

The downstream entry point is
`script.twd_tom.materialize_canonical_dataset`, which calls
`werewolf.offline_annotation` (C1) and
`werewolf.offline_materialization` (D), followed by
`script.twd_tom.split_offline_d_training_data`, `TWDToMDataset`, and the
current `script.twd_tom.train` / `eval` entry points. Online belief collection
and pre-D processing commands moved to `archive/legacy_tom` and are not normal
mainline entry points.

## External runtime storage

The following root directories are intentionally outside the Git delivery
boundary:

| Root | Typical contents |
|---|---|
| `data/` | Collection/source raw runs |
| `datasets/` | Identified dataset packages, aggregated raw data, materialized formal data, and frozen splits |
| `logs/` | Game, backend-call, configuration, and run logs |
| `outputs/` | Training/evaluation checkpoints and metrics |
| `review/` | Data audits, human review, reparse reports, and review packages |
| `checkpoints/` | Model checkpoints stored independently of a run tree |
| `models/` | Root-level external model weights or provider caches |

The root-level `models/` storage path is distinct from the tracked source
package `werewolf/models/`.

Small, sanitized examples should be added under a deliberate `examples/`
directory rather than relaxing the ignore policy for runtime data roots.

## Local-only files

Secrets, virtual environments, caches, IDE state, generated presentation
assets, and provider downloads are local-only. `.env.example` is the sole
tracked credential template; `.env` must never be committed.
