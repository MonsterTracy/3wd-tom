# Classic7 pre-speech ToM contract

## Research target

The ToM subsystem collects observer-specific suspicion in fixed-role,
seven-player Werewolf: two Werewolves, one Seer, one Witch, and three
Villagers. First-order ToM predicts a distribution over the 21 canonical
two-Werewolf pairs from public history plus the current observer's private
knowledge. Second-order ToM predicts, from public history alone, each modeled
observer's suspicion distribution over the seven canonical players. Dead
players remain valid identity candidates but do not produce reports. The
three-way decision subsystem is outside this contract.

## Time and information boundary

Each public speech follows:

`H_t -> B_t -> A_t -> H_{t+1}`

`H_t` is the already committed prefix of the sole append-only
`public_events` history (`classic7_public_event_sequence_v1`). It contains
public phase changes, turn starts, complete public speeches, revealed
voter-target results, exile results, and death announcements in publication
order. Before the current speaker generates `A_t`, every alive observer
reports from its own detached, legal observation. Reporting cannot mutate
agent memory, enter the current speaker prompt, or alter game state. The
current speech appears only in the next snapshot. System events do not
independently trigger snapshots, but are present before the next `turn_start`.

Each `public_speech` stores the final public `raw_text` and its exact
`sp_actions`. The latter is a structured speech summary, not the complete
`H_t`. The model projects every public event to structured tokens and does
not encode `raw_text`. `public_event_digest` covers the full canonical event
JSON including text; `structured_input_digest` covers exactly the raw-text-free
pre-token projection. Model features contain only that structured projection.
Reports, roles, private observations,
actual roles, teammate information, Seer checks, Witch knife targets, and all
future events are supervision-side data only.

## Hard knowledge

For each alive observer the environment deterministically stores closed sets
`known_werewolves` (`K+`) and `known_non_werewolves` (`K-`):

- Villager: `K-` contains self.
- Seer: `K-` contains self and good checks; wolf checks enter `K+`.
- Witch: `K-` contains self and every knife target legally seen while the
  antidote was unused. A saved target remains known good; after antidote use,
  later knife targets are not visible. Poison conveys no identity.
- Werewolf: `K+` is self plus teammate and `K-` is the other five players.
- Death and exile reveal no identity.

Let `Omega` be the single canonical ordering of the 21 unordered player
pairs. `Omega_hard` contains pairs that include all of `K+` and none of
`K-`. It must be nonempty. Exact-two-Werewolves closure adds to `K+` players
present in every pair and to `K-` players absent from every pair. No speech-
or probability-based inference is part of closure.

## Online raw collection

The current reporter collects a player-level suspicion set:

```json
{"suspected_werewolves":["player2","player5"]}
```

`suspected_werewolves` means the players this observer currently considers
relatively more suspicious, using only the frozen public history and its own
legal private information. It is not a complete two-Werewolf combination
constraint. A set of size one does not confirm one wolf, size two does not
assert the exact pair, and larger sets do not assert that both wolves are
inside the set. Empty, single-player, and multi-player sets are structurally
allowed.

Every `K+` player must be present and every `K-` player must be absent. IDs
are canonical, duplicate-free, and stored in player order. Let `E` be every
canonical player outside `K-`. With no extra soft suspicion, the sole
canonical report is `S = K+`; this is `[]` only when `K+` is empty. Reporting
`S = E` is invalid when `E != K+`, because it gives no relative preference.
When hard knowledge determines the complete candidate set,
`S = E = K+` remains valid. Invalid JSON, fields, IDs, hard-knowledge
conflicts, or noncanonical full-candidate reports fail closed without retry,
repair, fallback, truth injection, or reuse of an earlier report.

The current raw schema is
`classic7_pre_speech_player_suspicion_v2`; the prompt version remains
`classic7_pre_speech_player_suspicion_prompt_v2`. The previous v3
pair-support reporter remains only in Git history and has no runtime parser
or fallback. Raw JSONL stores `suspected_werewolves` and does not create pair
support, pair targets, or marginals.

## Audit-only offline pair projection

The sole current projected schema is
`classic7_pre_speech_suspicion_pair_distribution_v2`, produced explicitly
from the raw schema by projection version
`classic7_player_suspicion_pair_projection_base2_v1`.

For observer suspicion `S`, let `U = S \ K+`. For each pair `p` in
`Omega_hard`, the unnormalized weight is:

`w(p) = 2 ** |p intersect U|`

Thus a pair containing zero, one, or two soft-suspected players receives
weight `1`, `2`, or `4`. The version has no configurable beta. Suspicion
changes relative weight but never creates a hard zero; only `K+` and `K-`
exclude pairs. `K+` is removed from `U` because it is already enforced in
every hard-legal pair and must not be counted again as soft evidence.

Weights are normalized across `Omega_hard`. Empty suspicion beyond `K+`
therefore gives a uniform hard-legal distribution. One suspect gives
containing pairs twice the weight of other legal pairs. Two or more suspects
use the same per-pair hit count; they do not assert an exact pair or a hard
support. If every legal candidate is suspected, every legal pair has the
same hit count and the formula would be uniform, which is why raw v2
rejects that noncanonical form unless `S = E = K+`. The projection formula
and version are unchanged. Changing this formula requires a new projection
version and does not require recollecting raw suspicion data.

Projected JSONL retains the raw suspicion and audit metadata and adds
`pair_targets`, projection metadata, and deterministic-encoding flags.
Non-`ok` observers keep their status and error and receive a null target.
Projection and split utilities continue to validate that stored targets match
the declared projection. They are not the formal model-training input and do
not define the second-order output space.

The formal Dataset reads the current annotated split files directly. With
`--tom-order 1` it requires one current speaker observer, exposes only that
observer's two seven-player private-knowledge vectors, and projects the report
to the canonical 21 pair classes. With `--tom-order 2` it may supervise
multiple observers and exposes no private-knowledge model tensors. For each
valid second-order observer with suspicion set `S`, each player in a nonempty
`S` receives probability `1 / |S|`; an empty `S` becomes the uniform
seven-player distribution. The observer itself is not excluded. `known_*`
remains audit metadata and does not alter this target.

## Game-level dataset split

The splitter accepts one projected JSONL and partitions complete `game_id`
groups into `train.jsonl`, `validation.jsonl`, and `test.jsonl`. The caller
must provide positive train, validation, and test game counts whose sum
exactly equals the number of distinct games. A local seeded shuffle assigns
games deterministically; snapshots from one game never cross partitions, and
their input-relative order is preserved. Fixed twelve-game and 8/2/2 rules
are not built into the splitter; formal experiment counts must be supplied
explicitly. These projected splits remain available for audit; the
order-specific Qwen2 trainer does not consume them.

## Explicit stage pipeline

The standard user entry point is:

```bash
python -m script.twd_tom.pipeline \
  --config configs/twd_tom_pipeline_debug.yaml \
  --run-id debug4101 \
  --stage collect \
  --game-count 1 \
  --seeds 4101
```

Supported stages are `validate`, `collect`, `project`, and `split`. Each stage
must be invoked explicitly; the pipeline only validates configuration and
calls the existing non-training stage function. Configuration stores stable
runtime, budget, schema, and split parameters.
The required CLI `run_id` identifies one explicit run; optional
`--game-count` and `--seeds` jointly override only `validate` or `collect` for
the current process. Repeated seeds are allowed and preserve their order.

Artifacts for one `run_id` are derived without scanning for a latest run:
game logs, call audit, manifest, and resolved config are under
`logs/tom/<run_id>/`; raw, projected, and split data are under
`data/tom/<run_id>/`. The `outputs/tom/<run_id>/` path remains reserved by the
runtime path contract but the non-training pipeline does not write model
artifacts there. The resolved config is written to the log directory and
never back to `configs/`; no timestamp config is generated. Secrets remain in
environment variables loaded from `.env`, never in YAML or stage summaries.
Raw and projected artifacts remain distinct. The separate Qwen2 training
entry selects one current annotated raw file by `tom_order` and writes
checkpoints below a `tom_order_1/` or `tom_order_2/` output directory.
Three-way decision is not implemented.

Pipeline configuration fixes the public-event, raw, projected, and projection
versions above. Earlier three-game artifacts remain audit-only and are not
training input. Whether future models should encode `raw_text` is deferred to
a later structured-input collision study.

The model input is the structured `public_events` projection. Its sole
backbone is a randomly initialized Hugging Face `Qwen2Model` receiving
`inputs_embeds`; it never loads pretrained weights or uses a tokenizer. The
last non-padding event state is added to each observer embedding. First-order
rows also add one linear projection of that observer's private knowledge and
the single output projection produces `[B,7,21]` pair logits. Second-order rows
remain public-only and their order-specific output projection produces
`[B,7,7]` player-suspicion logits. Both orders use the same masked soft-target
categorical cross entropy. Three-way decision remains outside this contract.
