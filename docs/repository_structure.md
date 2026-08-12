# Repository structure and storage boundary

## Source tree

| Path | Responsibility |
|---|---|
| `werewolf/agents/` | Agent abstractions, Planner/Renderer orchestration, prompt and action contracts |
| `werewolf/backends/` | Named OpenAI-compatible backend loading and request handling |
| `werewolf/envs/` | Seven-player game state, observations, valid actions, phases, and public events |
| `werewolf/speech/` | Online/offline speech parsing and playing-agent belief reports |
| `werewolf/models/twd_tom/` | ToM schemas, collection helpers, Dataset, backbones, targets, losses, metrics, and shadow inference |
| `script/twd_tom/` | Collection, projection, split, audit, training, and evaluation entry points |
| `tests/` | Self-contained deterministic tests organized by subsystem |
| `configs/` | Reusable runtime profiles; see `configs/README.md` |
| `docs/` | Research contracts, architecture, provenance, and deployment guidance |

The root entry points `run_battle.py`, `run_random.py`, and `run_batch.sh`
remain part of the gameplay interface. The repository does not contain a
`jobs/` directory or an in-repository model-service manager.

## Collection entry points

`python -m script.twd_tom.pipeline --stage collect` is the canonical user
interface for formal collection. It validates one named run and calls the
audited per-game collection core.

The other maintained modules are deliberately narrower:

- `script.twd_tom.collect`: one game and one explicit raw sample path;
- `script.twd_tom.formal_batch_collection`: monitored ten-game batch utility;
- `script.twd_tom.real_backend_dry_run`: bounded two-game audit CLI and shared
  audited per-game core;
- `script.twd_tom.monitored_collection`: internal monitored batch runner;
- `script.twd_tom.reparse_speeches`: offline speech reparse audit.

They are not automatically chained and are not fallback implementations.

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
