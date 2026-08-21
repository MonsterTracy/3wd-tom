# Server deployment checklist

1. Clone the repository with Git; do not copy the complete local working tree.
2. Check out and record the intended Git commit.
3. Create an isolated Python environment on the server.
4. Install the project with `pip install -e .`.
5. Run `cp .env.example .env`, then fill only the required server-side values.
6. Never commit or transfer the workstation `.env`.
7. Download or mount model weights independently on the server.
8. Keep model weights outside the Git repository.
9. Start an OpenAI-compatible inference service supported by the server.
10. Set the correct `base_url` and model ID in a reusable config.
11. Prefer loopback when the model service and project run on the same host.
12. Loopback endpoints automatically use `trust_env=False`.
13. Verify the endpoint's `/v1/models` response.
14. Verify one bounded `/v1/chat/completions` request.
15. Run `python -m pytest -q tests/runtime/test_local_mlx_config.py`.
16. Run the full suite with `python -m pytest -q`.
17. Choose a new, unique canonical trajectory `run_id` and an absent A/C0 output directory.
18. Run `python -m script.twd_tom.collect_canonical_trajectories` with explicit config, seed range, game count, and output root.
19. Ensure the external trajectory, annotation, dataset, and output roots are writable.
20. Put runtime data, logs, and checkpoints on persistent server storage.
21. Do not commit runtime artifacts to Git.
22. Produce C1 only from validated paired A/C0 artifacts, then materialize D.
23. Run `script.twd_tom.split_offline_d_training_data` before training.
24. Ensure `train` never reads the test split.
25. Read the test split only during `eval`.
26. Back up `raw.jsonl`, `projected.jsonl`, the split and run manifests, call
    audit, checkpoints, and metrics.
27. After failure, use a new `run_id`; do not overwrite the failed run.
28. Manage server restarts and model processes with an external service manager.
29. Do not add automatic model downloads, retries, or provider fallback.
30. Record the Git commit, config file, `run_id`, model ID, endpoint, seeds, and
    environment versions for every deployment.
