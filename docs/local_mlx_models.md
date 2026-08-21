# Local MLX model endpoints

The local MLX runtime uses the existing OpenAI-compatible backend interface.
The project does not download models or start and stop these services.

Start the three independent servers in separate terminals:

```bash
conda activate mlxserver

python -m mlx_lm.server \
  --model "mlx-community/Qwen2.5-7B-Instruct-4bit" \
  --host 127.0.0.1 \
  --port 8080

python -m mlx_lm.server \
  --model "mlx-community/Llama-3.1-8B-Instruct-4bit" \
  --host 127.0.0.1 \
  --port 8081

python -m mlx_lm.server \
  --model "mlx-community/Mistral-7B-Instruct-v0.3-4bit" \
  --host 127.0.0.1 \
  --port 8082
```

Use [`configs/twd_tom_local_mlx.yaml`](../configs/twd_tom_local_mlx.yaml).
It defines these exact aliases:

| Alias | Base URL | Model |
|---|---|---|
| `local_qwen25_7b` | `http://127.0.0.1:8080/v1` | `mlx-community/Qwen2.5-7B-Instruct-4bit` |
| `local_llama31_8b` | `http://127.0.0.1:8081/v1` | `mlx-community/Llama-3.1-8B-Instruct-4bit` |
| `local_mistral7b_v03` | `http://127.0.0.1:8082/v1` | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` |

The three agent profiles form an equal-weight random pool and are not bound to
roles or factions. Each playing agent's belief report uses that same agent's
backend alias and model. The parser is explicitly routed to
`local_qwen25_7b`.

For single-model collection, use
[`configs/twd_tom_local_qwen25_7b.yaml`](../configs/twd_tom_local_qwen25_7b.yaml)
and start only the Qwen server on port 8080. Gameplay, belief reporting, and
parsing all use `local_qwen25_7b`. The three-model configuration remains
available for later mixed-model experiments.

Explicit loopback endpoints may omit `api_key_env`; the client supplies a
non-secret placeholder required by the OpenAI SDK. Remote endpoints and local
endpoints that explicitly name `api_key_env` keep the existing fail-closed
environment-variable requirement.

For `localhost`, the `127.0.0.0/8` range, and `::1`, the OpenAI-compatible
client explicitly ignores environment proxy settings. Remote and cloud
endpoints retain the SDK's default proxy behavior, so users do not need to set
`NO_PROXY` manually. This does not change retry or fail-closed behavior.

The historical V2.7 validator remains available in the archive and does not
contact a model service:

```bash
python -m archive.legacy_tom.script.twd_tom.pipeline \
  --config configs/twd_tom_local_mlx.yaml \
  --run-id local_mlx_check \
  --stage validate
```
