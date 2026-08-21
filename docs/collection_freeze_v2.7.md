# Collection Freeze V2.7

> Historical contract: its executable modules now live under
> `archive.legacy_tom`. It is retained for reproduction and is not the current
> A/C0 -> C1 -> D collection mainline.

## Frozen source

- Freeze commit: `844784d7232af7e40633b62c450f72e4c35edb8e`
- Canonical config: `configs/twd_tom_server_qwen35_9b.yaml`
- Historical entry: `python -m archive.legacy_tom.script.twd_tom.pipeline --stage collect`

The source commit identifies the collection generator. Presentation-only
documentation or repository-layout changes after this commit do not define a
new collection freeze.

## Freeze pilot

- Seeds: 555–574
- Games: 20
- Execution: four independent five-game runs
- Completion: 20 / 20 games
- Final audit: PASS

Each run requires its own unique `run_id`. The run manifest and resolved
configuration stored with the external artifacts are authoritative for the
exact run IDs and invocation details.

## Canonical invocation

Validate each five-seed run before collection:

```bash
python -m archive.legacy_tom.script.twd_tom.pipeline \
  --config configs/twd_tom_server_qwen35_9b.yaml \
  --run-id <UNIQUE_RUN_ID> \
  --stage validate \
  --game-count 5 \
  --seeds <FIVE_SEEDS>

python -m archive.legacy_tom.script.twd_tom.pipeline \
  --config configs/twd_tom_server_qwen35_9b.yaml \
  --run-id <UNIQUE_RUN_ID> \
  --stage collect \
  --game-count 5 \
  --seeds <FIVE_SEEDS>
```

The four seed groups are 555–559, 560–564, 565–569, and 570–574. The pipeline
does not automatically run projection, split, training, or evaluation.

## Frozen contract chain

- **V2.2 — true-Werewolf self-disclosure:** a true Werewolf cannot represent
  a public plan that explicitly identifies the speaker as a Werewolf.
- **V2.3 — PK self-vote exclusion:** each PK voter's candidate list excludes
  that voter while preserving the canonical PK pool.
- **V2.4 — exact duplicate canonicalization:** identical public action-target
  pairs are stably deduplicated without merging distinct actions.
- **V2.5 — night action fidelity:** the environment executes the night action
  selected by the model without index remapping.
- **V2.6 — structured night selection:** night choices use a constrained,
  structured response contract tied to the canonical valid-action list.
- **V2.7 — public-speech reference normalization:** deterministic first-person
  and grouped player references satisfy plan-target coverage without semantic
  guessing.

These labels summarize frozen runtime contracts; they are not a development
changelog and do not authorize fallback, retry, or automatic correction.

## Artifact boundary and provenance

The Git repository does not contain raw data, game logs, call-audit logs,
review archives, model weights, or checkpoints. Those artifacts belong to
external runtime storage.

Formal data provenance must be recoverable from all of the following:

1. `source_commit` matching the freeze commit above;
2. the per-run manifest;
3. the resolved runtime configuration;
4. the exact seed range;
5. artifact hashes maintained with the external data.

No workstation path, server username, private host, credential, or API token
is part of this repository-level provenance record.
