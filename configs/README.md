# Canonical configurations

The repository tracks two secret-free Qwen runtime profiles. Both connect
gameplay, readonly belief reporting, and speech parsing to the `qwen3.5-9b`
OpenAI-compatible service on `127.0.0.1:8000`; the repository does not start or
download that model.

- `twd_tom_server_qwen35_9b.yaml`: three-game pilot with seeds 4101--4103.
- `twd_tom_server_qwen35_9b_canonical_60.yaml`: a predeclared 120-seed pool
  (4201--4320) that stops after 60 successful canonical games and uses a
  frozen 48/6/6 game-level split declaration.
- `onuw_parity_pilot_qwen35_9b.yaml`: three entirely new parity pilot games
  (seeds 5101--5103). It is fail-closed, uses strict PRE full role guesses and
  agent-declared face/tone, and is explicitly ineligible for formal training.

Use these profiles through the explicit-stage pipeline. The collector
consumes and enforces every `pipeline.collection` field: the CLI seed range and
game count must match the YAML exactly, and category/total call budgets plus the
per-game wall-clock budget are active runtime limits. The frozen contract is
recorded in `docs/collection_contract.md`.
