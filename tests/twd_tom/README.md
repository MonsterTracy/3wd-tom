# TWD-ToM tests

This directory groups the active TWD-ToM contract tests:

- schema and hard-knowledge validation;
- pre-speech belief snapshots and label semantics;
- public-event collection and encoding;
- suspicion-to-pair projection;
- formal splitting and Dataset normalization;
- feature encoding, model shape, loss, and metrics;
- training and evaluation entry points;
- pipeline orchestration and synthetic collection.

Backbone tests exercise the sole, randomly initialized Hugging Face
`Qwen2Model` path, including `inputs_embeds`, causal and padding behavior,
order-specific private inputs, and strict checkpoint restoration.
