# Archived legacy ToM implementations

This directory preserves research code that predates the canonical pipeline:

```text
trajectory A/C0
  -> offline annotation C1
  -> offline materialization D
  -> split_offline_d_training_data
  -> TWDToMDataset
  -> train/eval
```

The archive is importable so historical tests and explicitly requested
reproduction runs remain possible. It is not imported by canonical C1/D,
Dataset, training, or evaluation code.

## Contents

- `script/tom/`: the earlier formal-ToM pilot splitter and trainer;
- `werewolf/models/tom/`: its collector, schema, event projection, Dataset,
  model, target, and loss implementation;
- `script/twd_tom/`: online V2.7 belief collection, projection,
  materialization, and split utilities.

Historical module commands use the archive prefix, for example:

```bash
python -m archive.legacy_tom.script.tom.train --help
python -m archive.legacy_tom.script.twd_tom.pipeline --help
```

The optional legacy formal collector hook in `run_random.py` also resolves its
collector/reporter from this archive. That preserves explicit historical use;
canonical trajectory collection passes only `trajectory_recorder`.

Deterministic legal-observation and hard-knowledge derivation moved to
`werewolf/observer_knowledge.py` because canonical offline annotation still
owns that contract. The archived reporter delegates to the same functions, so
the two paths cannot silently diverge.

No persisted schema, action semantics, Environment behavior, or dataset target
meaning was changed by the archive migration.
