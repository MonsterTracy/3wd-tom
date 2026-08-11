# Configuration profiles

All tracked YAML files are secret-free runtime profiles. Credentials, when
required, are named through `api_key_env` and remain in the process environment
or local `.env`. Loopback endpoints are deployment profiles, not embedded
model services; this repository does not start or download those models.

## Canonical Collection V2.7

| Config | Purpose |
|---|---|
| `twd_tom_server_qwen35_9b.yaml` | Canonical Collection V2.7 profile: one `qwen3.5-9b` OpenAI-compatible service on `127.0.0.1:8000`, strict gameplay speech, JSON Schema support, and explicit call budgets |

Use this profile through the explicit-stage pipeline. The freeze provenance is
recorded in `docs/collection_freeze_v2.7.md`.

## Qwen2.5 profiles

| Config | Purpose |
|---|---|
| `twd_tom_server_qwen25_7b.yaml` | Server-side single Qwen2.5-7B endpoint on port 8000 with strict gameplay and bounded generation |
| `twd_tom_local_qwen25_7b.yaml` | Local single Qwen2.5-7B MLX endpoint on port 8080 |

These profiles are distinct model/runtime variants and are not aliases for the
V2.7 Qwen3.5 freeze.

## Local multi-model profile

| Config | Purpose |
|---|---|
| `twd_tom_local_mlx.yaml` | Local MLX pool containing Qwen2.5-7B, Llama-3.1-8B, and Mistral-7B endpoints on ports 8080–8082 |

See `docs/local_mlx_models.md` for the corresponding local service commands.

## Gameplay and baseline profiles

| Config | Purpose |
|---|---|
| `deepseek_vs_twdm.yaml` | DeepSeek village team against a local Qwen2.5-14B TWDM Werewolf profile |
| `gpt_vs_twdm.yaml` | Local Qwen2.5 gameplay profile against a local Qwen2.5-14B TWDM Werewolf profile; despite the historical filename, it uses the generic GPT-style agent class rather than an OpenAI GPT model |
| `random_models.yaml` | Random competition pool across DeepSeek, local Qwen2.5, and local TWDM profiles |

## Debug and smoke profiles

| Config | Purpose |
|---|---|
| `twd_tom_pipeline_debug.yaml` | Three-game DeepSeek pipeline validation/debug profile |
| `twd_tom_deepseek_only_debug.yaml` | Seven nominal DeepSeek candidate profiles for deterministic assignment/debug checks |
| `tom_belief_smoke.yaml` | Parser and environment smoke profile without action-agent construction |

These profiles are useful for validation and regression work; they do not
define a formal collection freeze.

## Legacy / purpose needs confirmation

| Config | Status |
|---|---|
| `twd_tom_collect.yaml` | Earlier mixed-API collection profile with a fixed village-team profile plus a random candidate pool; retain until historical run provenance is checked |
| `twd_tom_multi_api.yaml` | Near-duplicate mixed-API pool without the fixed village-team section; retain until its independent experimental purpose is confirmed |

No YAML file is moved or renamed during presentation cleanup because config
paths may be recorded in external run manifests.
