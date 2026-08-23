# Configuration profiles

All tracked YAML files are secret-free runtime profiles. Credentials, when
required, are named through `api_key_env` and remain in the process environment
or local `.env`. Loopback endpoints are deployment profiles, not embedded
model services; this repository does not start or download those models.

## tom-v2 canonical collection

| Config | Purpose |
|---|---|
| `twd_tom_server_qwen35_9b.yaml` | Canonical tom-v2 profile: one `qwen3.5-9b` OpenAI-compatible service on `127.0.0.1:8000`, strict gameplay speech, JSON Schema support, and explicit call budgets |

Use this profile through the explicit-stage pipeline. The collector now
consumes and enforces every `pipeline.collection` field: the CLI seed range and
game count must match the YAML exactly, and category/total call budgets plus the
per-game wall-clock budget are active runtime limits. The frozen contract is
recorded in `docs/collection_contract.md`.

## Qwen2.5 profiles

| Config | Purpose |
|---|---|
| `twd_tom_server_qwen25_7b.yaml` | Server-side single Qwen2.5-7B endpoint on port 8000 with strict gameplay and bounded generation |
| `twd_tom_local_qwen25_7b.yaml` | Local single Qwen2.5-7B MLX endpoint on port 8080 |

These profiles are distinct model/runtime variants and are not aliases for the
canonical Qwen3.5 profile.

## Local multi-model profile

| Config | Purpose |
|---|---|
| `twd_tom_local_mlx.yaml` | Local MLX pool containing Qwen2.5-7B, Llama-3.1-8B, and Mistral-7B endpoints on ports 8080–8082 |

See `docs/local_mlx_models.md` for the corresponding local service commands.

## Debug and smoke profiles

| Config | Purpose |
|---|---|
| `twd_tom_pipeline_debug.yaml` | Three-game DeepSeek pipeline validation/debug profile |
| `twd_tom_deepseek_only_debug.yaml` | Seven nominal DeepSeek candidate profiles for deterministic assignment/debug checks |
| `tom_belief_smoke.yaml` | Parser and environment smoke profile without action-agent construction |

These profiles are useful for validation and regression work; they do not
define a formal collection freeze.
