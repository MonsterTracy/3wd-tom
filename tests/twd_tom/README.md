# TWD-ToM tests

This directory groups the active TWD-ToM contract tests:

- schema and hard-knowledge validation;
- pre-speech belief snapshots and label semantics;
- public-event collection and encoding;
- deterministic suspicion-set conversion;
- canonical game-level splitting and Dataset normalization;
- feature encoding, model shape, loss, and metrics;
- training and evaluation entry points;
- canonical collection and materialization entry points.
- enforced collection call/wall-clock budgets and pre-materialization canonical audit.
- soft-target top-1 support-hit semantics and uniform non-self baselines.

Backbone tests exercise the sole, randomly initialized Hugging Face
`Qwen2Model` path, including `inputs_embeds`, causal and padding behavior,
order-specific private inputs, and strict checkpoint restoration.
