# Canonical configuration

The repository tracks one secret-free runtime profile:
`twd_tom_server_qwen35_9b.yaml`. It connects gameplay, readonly belief
reporting, and speech parsing to the `qwen3.5-9b` OpenAI-compatible service on
`127.0.0.1:8000`. The repository does not start or download that model.

Use this profile through the explicit-stage pipeline. The collector
consumes and enforces every `pipeline.collection` field: the CLI seed range and
game count must match the YAML exactly, and category/total call budgets plus the
per-game wall-clock budget are active runtime limits. The frozen contract is
recorded in `docs/collection_contract.md`.
