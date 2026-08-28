# Frozen one-shot sealed evaluation

`script.twd_tom.run_sealed_eval` is the only formal sealed-test entry point for
the frozen `development_final_fit_v1` checkpoint. It is intentionally separate
from the training and general evaluation CLIs.

## Fixed contract

The command accepts only five runtime inputs: checkpoint path, final protocol
path, split manifest path, new output directory, and device. The model,
features, annotations, supervision scope, epoch, batch size, bootstrap seed,
and metric definitions are constants in the evaluator and cannot be overridden
from the command line.

Before opening the test JSONL, the evaluator verifies the exact checkpoint hash,
final-protocol digest, training Git commit, checkpoint/task fields, finite
checkpoint tensors, manifest lineage, the exact 54/6 disjoint game partition,
a clean committed evaluator worktree, a new output directory, and the absence
of the checkpoint's permanent one-shot marker. The sealed label file is not
needed for this preflight.

After preflight, it freezes `sealed_test_protocol.json` and atomically creates
`<checkpoint>.sealed_test_evaluated.json`. The marker is checkpoint-scoped, so
changing the output directory does not permit a second forward pass. A failed
run after marker creation remains locked and requires a separately reviewed
recovery policy; there is no automatic retry or fallback.

## Evaluation and artifacts

Evaluation uses `model.eval()` under `torch.inference_mode()`. It creates no
optimizer or scheduler, calls no backward pass, performs no selection or
tuning, and never writes a checkpoint. It uses dense V1 public speech,
`v1_empty_unobserved` belief labels, the `no_phase_day` input profile, and
`non_wolf_alive` supervision. Empty/unobserved rows remain zero targets and are
excluded from all distribution metrics.

One successful run places exactly these JSON artifacts in its output directory:

- `sealed_test_protocol.json`: immutable pre-label-open protocol and digest.
- `sealed_test_summary.json`: primary non-wolf result, secondary metrics, and
  Villager diagnostic.
- `sealed_test_per_game.json`: additive per-game KL terms and per-game
  GapClosed values used for metric-only recomputation.
- `sealed_test_provenance.json`: input/output hashes and pure-forward audit.

The checkpoint-scoped permanent marker is stored beside the checkpoint, outside
that output directory.

The primary estimand is the common scored-game macro mean of
`1 - model_kl_sum / uniform_kl_sum`. The report also records the independently
computed observer-weighted ratio, both KL sums and means, total variation,
observed-row and scored-game counts, per-game values, and a game-unit percentile
bootstrap with 2,000 draws and seed 42. With six sealed games, the bootstrap is
reported descriptively and is not presented as a precision claim.

Formal sealed execution is intentionally not documented here as a copy-paste
command until the implementation commit and review are approved.
