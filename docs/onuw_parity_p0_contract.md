# ONUW-parity P0 frozen contract

This document separates literal public-code behavior from the minimal
Classic-7 adaptation and from later Classic-7 research improvements. It is
the implementation contract for branch `onuw-parity`, based on `ccbc15b`.

## Protocol layers

| Dimension | `ONUW-code-reference` | `Classic7-ONUW-reference` | `Classic7-main` |
|---|---|---|---|
| Purpose | Record public code as it behaves, including quirks | Minimal functional adaptation to strict Classic-7 | Later semantic improvements and ablations |
| Players | 5 | 7 | 7 |
| Timing | Public-code timing, including PRE/current-token mismatch | Strict PRE | Strict PRE unless separately declared |
| Canonical training unit | Full game action sequence | Full game public token sequence with multiple PRE queries | Must declare explicitly |
| Label report | Full role guesses; extract guessed-werewolf support | `onuw_style_role_guess`; full seven fields, legal role vocabulary | Direct suspicion or other declared collector |
| Empty support | Uniform over all players | Uniform over all seven players | Legal-candidate uniform is an adaptation |
| Self | Included | Included | Diagonal mask is an adaptation |
| Observer rows | All public-code observers | Alive observers only | Must declare |
| Target columns | All players | All seven canonical players, including dead players | Dead-target masking is an adaptation |
| Collector information | ONUW agent state | Observer's legally visible public and private view | Must declare |
| Model input | Public structured actions | Public-only | Private conditioning is an adaptation |
| Input profile | subject/action/object/tone/face | `onuw_action_only` or `classic7_public_events` | Must declare |
| Emotion | Agent-declared 8-class face and tone | Full: agent-declared 8-class; ablation: remove/zero contribution | Must declare |
| Backbone | 8 GPT2Block, 512 hidden, 8 heads | 8 GPT2Block, 512 hidden, 8 heads | Qwen2 etc. are adaptations |
| Readout | Direct full matrix | Hidden at each PRE query -> direct 49-value matrix head | Observer-query is an adaptation |
| Objective | Public code KL | Row-micro soft-target CE; report `KL=CE-H(target)` | Alternative weighting is an adaptation |

`other` is one real emotion class. Missing emotion is never encoded as
`other`. `onuw_no_face_no_tone` removes the face/tone contribution by emitting
zero IDs.

## Label authority contract

The target is private-informed, while the learned model input is public-only.
The role-guess collector:

- must use the observer's detached legal observation, including that
  observer's own role and legally visible private events;
- must not receive a global oracle role map, another player's invisible
  private state, or the true two-wolf constraint;
- hard-validates exactly `player1` through `player7` and the legal role
  vocabulary;
- only audits role-count conflict; it never rejects, repairs, renormalizes, or
  uses actual roles to make a report satisfy the true composition;
- converts every guessed-as-werewolf player to uniform support; three guessed
  wolves therefore produce three entries of `1/3`;
- converts empty support to uniform `1/7` across all players.

The existing `suspected_werewolves` collector remains an adaptation/ablation
and is not called ONUW-equivalent.

## Sequence and death contracts

The canonical item is one game. It contains one chronological token sequence
and multiple strict PRE queries. `token_cutoff` is the last visible token at a
query and may repeat when intervening speech produces zero structured actions.
An explicit BOS token makes an empty public history queryable. Sparse
single-PRE prefixes may be materialized only for debug/evaluation.

Only alive observer rows are supervised. Every supervised row always has
seven columns in canonical seat order, including dead players and self. P0
has no dead-target, diagonal, or hard-knowledge candidate mask.

A batch has three distinct masks:

- `token_attention_mask: [B,L]`
- `query_valid_mask: [B,Q]`
- `observer_alive_mask: [B,Q,7]`

They are not aliases and do not share semantics.

## Loss and reporting

For every valid `(game, query, alive observer)` row:

```text
CE = -sum(target * log_softmax(logits))
H(target) = -sum(target * log(target))
KL(target || prediction) = CE - H(target)
```

The optimization loss is the mean CE over all valid observer rows in the
batch (`observer-row-micro`). Evaluation also reports row-micro, query-macro,
and game-macro CE and KL. There is no target-column mask.

The ONUW paper training record is retained as historical evidence, not as the
final Classic-7 default:

```text
epochs=80
batch_size=32
learning_rate=5e-5
early_stopping=validation_loss
```

Because ONUW's public Dataset item and this game-with-many-queries item imply
different optimizer-step budgets, the Classic-7 training budget remains
unfrozen until pilot statistics are reviewed.

## Required pilot audits

Before choosing `max_positions`, report sequence length p50/p90/p95/p99/max
and the longest PRE query prefix. The dataset never truncates. If a required
length exceeds 256, record a Classic-7 capacity adaptation and choose the
capacity explicitly.

For `onuw_action_only`, report:

- zero-action speech rate;
- consecutive PRE queries sharing `token_cutoff`;
- same-context/different-target rate;
- mean TV and JS between adjacent targets sharing token context.

These measurements record structured-Perceiver information loss and do not
change the P0 input.

After the pilot is materialized, `profile_onuw_parity_memory.py` runs an
actual CUDA AdamW forward/backward/step at an explicit game batch size and
reports peak allocated/reserved bytes. It uses the audited maximum sequence
length and refuses to truncate; this measurement informs, but does not itself
freeze, the formal training budget.

## Paper/code discrepancy record

| Item | Paper description | Public code behavior | P0 decision |
|---|---|---|---|
| Label elicitation | Direct suspicion after speech | Full role guesses, then extract wolves | Preserve code-derived collector, but place it at strict PRE |
| Loss wording | Cross entropy | KL divergence | Optimize equivalent soft-target CE and report KL |
| Sample unit | Described at experiment level | One full JSON/game action sequence | One game sequence with multiple PRE query positions |
| Emotion | Agent-generated face/tone | subject/action/object/tone/face tensors | Full declared multimodal reference plus a true removal ablation |
| Empty report | Not fully specified | Uniform over all players | Full uniform over all seven players |
| Timing | Belief around action generation | Public code has label/current-token mismatch | Do not copy the mismatch; strict PRE is mandatory |
| Padding/counts | Not a semantic contribution | Public implementation quirks exist | Record literal quirks only under code reference; do not repair that reference |

## Non-destructive lineage

- `f3f1d0a` is archived by annotated tag
  `classic7-semantic-clean-v1`.
- Branch `onuw-parity` and its independent worktree start at `ccbc15b`.
- Existing semantic-clean code and the six-game sealed test lineage are not
  reset, rewritten, selected against, or reused.
- New parity pilots require a new namespace and new development/test/sealed
  boundary.

P0 keeps the Classic-7 environment, strict PRE snapshot machinery, canonical
public event/audit infrastructure, and game-level split guarantees from the
starting commit. It adds isolated parity role-guess, game-sequence, model,
loss, and audit modules. Qwen2, observer-query, `non_wolf_alive`, private model
conditioning, legal-candidate empty conversion, and diagonal/hard-knowledge
masks remain outside the P0 reference.

## P0 acceptance tests

1. Exact seven-field role reports and vocabulary are hard validated.
2. Three-wolf conflict is accepted and audited; actual roles cannot repair it.
3. Empty becomes full `1/7`; self and dead target columns remain legal.
4. Repeated PRE token cutoffs are valid and sparse prefixes are not the
   training unit.
5. Full multimodal rejects missing emotion; text-only produces zero emotion
   IDs rather than `other`.
6. Batches expose three separate masks.
7. Reference architecture is 512/8/8 with a direct 49-value head.
8. Loss is observer-row-micro, with query/game macro and KL reporting.
9. Capacity overflow raises instead of truncating.
10. A deterministic synthetic batch completes forward, loss, and backward.
