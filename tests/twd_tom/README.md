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
`GPT2Model` path, including causal and padding behavior, strict checkpoint
restoration, and rejection of the removed torch-Transformer architecture.
