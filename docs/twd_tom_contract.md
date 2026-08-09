# Classic7 pre-speech ToM contract

## Research target

The ToM subsystem collects observer-specific suspicion in fixed-role,
seven-player Werewolf: two Werewolves, one Seer, one Witch, and three
Villagers. First-order ToM predicts a distribution over the 21 canonical
two-Werewolf pairs from public history plus the current observer's private
knowledge. Second-order ToM predicts, from public history alone, each modeled
observer's internal distribution over the same 21 two-Werewolf worlds. The
two orders differ only in model input scope, not native belief space. Dead
players remain valid identity candidates but do not produce reports. The
three-way decision subsystem is outside this contract.

## Time and information boundary

Each public speech follows:

`H_t -> B_t -> A_t -> H_{t+1}`

`H_t` is the already committed prefix of the sole append-only
`public_events` history (`classic7_public_event_sequence_v2`). It contains
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
The v2 speech-action contract keeps the original seven action IDs and appends
`check_as_good`, `check_as_werewolf`, `save`, `poison`, `guard`, and
`vote_intent`, all in the same `[speaker, action, target_player]` form.
Reports, roles, private observations,
actual roles, teammate information, Seer checks, Witch knife targets, and all
future events are supervision-side data only.

### Speech action semantic modules and non-redundancy

The 13 actions remain one flat discrete vocabulary. The semantic modules are
prompt-only organization and do not enter events, features, embeddings, or
checkpoints: `ROLE_ESTIMATE` contains the five `point_as_*` actions;
`SOCIAL_STANCE` contains `support` and `oppose`; `CLAIMED_SKILL_REPORT`
contains `check_as_good`, `check_as_werewolf`, `save`, `poison`, and `guard`;
and `ACTION_INTENT` contains `vote_intent`. Under A1, extraction records only
explicit atomic propositions, uses the most specific action without expanding
it into broader correlated actions, and retains multiple actions only when the
speech explicitly states multiple independent propositions in source order.

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
multiple observers and exposes no private-knowledge model tensors. Each valid
second-order observer's report is projected to the same canonical 21 pair
classes using that observer's `known_werewolves` and
`known_non_werewolves`. Those fields are supervision-side target-construction
and audit data only: they do not appear in a second-order batch's model inputs,
do not enter `forward`, and do not mask logits.

The sole formal second-order supervision boundary is
`post_completed_public_speech_pre_next_action_v1`. The final two public events
must be one complete `public_speech`, including its stored `sp_actions`, and
the next reasoning player's `turn_start`; that reasoning player is the
sample's canonical `speaker_id`. Vote, phase, exile, death, and other
system-only boundaries are excluded.

The supervision scope is `all_valid_other_observers`: the effective mask is
the existing label-valid `subject_mask` intersected with canonical observer
IDs other than the reasoning player. The latest speech actor and nested
speech-action subjects do not select the supervised rows. Snapshots whose
effective mask is empty are excluded by the same deterministic Dataset indices
in train, validation, and eval. Target rows remain unchanged, and first-order
masking is unchanged.

## Audit-only projected dataset split

The splitter accepts one projected JSONL and partitions complete `game_id`
groups into `train.jsonl`, `validation.jsonl`, and `test.jsonl`. The caller
must provide positive train, validation, and test game counts whose sum
exactly equals the number of distinct games. A local seeded shuffle assigns
games deterministically; snapshots from one game never cross partitions, and
their input-relative order is preserved. Fixed twelve-game and 8/2/2 rules
are not built into the splitter; formal experiment counts must be supplied
explicitly. These projected splits remain available for audit; the
order-specific trainer does not consume them.

`script/twd_tom/split_formal_dataset.py` implements this audit split. It is
not the formal training-data splitter or a training entry point.

## Formal training operations

Formal first- and second-order data reuse one in-memory, seed-42 game split.
Generate the six order-specific files without overwriting existing outputs:

```bash
python -m script.twd_tom.split_training_data \
  --tom1 data/qwen25/raw_tom.jsonl \
  --tom2 data/qwen25/raw_tom2.jsonl \
  --output-dir data/qwen25 \
  --seed 42
```

The only formal split files are:

- `data/qwen25/tom1/train.jsonl`
- `data/qwen25/tom1/val.jsonl`
- `data/qwen25/tom1/test.jsonl`
- `data/qwen25/tom2/train.jsonl`
- `data/qwen25/tom2/val.jsonl`
- `data/qwen25/tom2/test.jsonl`

Train the two orders independently. The selected output root must be inside
the repository's logical path; its `tom_order_1/` or `tom_order_2/` run
directory must be absent or empty:

```bash
python -m script.twd_tom.train \
  --dataset data/qwen25/tom1/train.jsonl \
  --validation-dataset data/qwen25/tom1/val.jsonl \
  --tom-order 1 \
  --output-dir outputs/tom/formal_tom1 \
  --device auto \
  --seed 42

python -m script.twd_tom.train \
  --dataset data/qwen25/tom2/train.jsonl \
  --validation-dataset data/qwen25/tom2/val.jsonl \
  --tom-order 2 \
  --output-dir outputs/tom/formal_tom2 \
  --device auto \
  --seed 42
```

Training refuses a dirty Git worktree. `best.pt`, `last.pt`, `summary.json`,
and `history.json` record one `run_provenance` containing the commit SHA,
clean-worktree assertion, repository-relative train/validation paths and
their SHA-256 digests, Python/PyTorch/Transformers/platform versions,
requested and resolved devices, deterministic-algorithm status, and seed.
Python, Torch, CUDA, the DataLoader generator, and cyclic rotation are seeded;
supported Torch backends use deterministic algorithms and deterministic
cuDNN settings. These controls aim to reproduce a run with the same commit,
data, and environment. They do not promise bit-identical results across CPU,
CUDA, and MPS hardware.

Evaluate validation data against the exact hashed training data recorded by
the checkpoint:

```bash
python -m script.twd_tom.eval \
  --checkpoint outputs/tom/formal_tom1/tom_order_1/best.pt \
  --dataset data/qwen25/tom1/val.jsonl \
  --training-dataset data/qwen25/tom1/train.jsonl \
  --output outputs/tom/formal_tom1/tom_order_1/val_metrics.json \
  --device auto

python -m script.twd_tom.eval \
  --checkpoint outputs/tom/formal_tom2/tom_order_2/best.pt \
  --dataset data/qwen25/tom2/val.jsonl \
  --training-dataset data/qwen25/tom2/train.jsonl \
  --output outputs/tom/formal_tom2/tom_order_2/val_metrics.json \
  --device auto
```

After the model definition, hyperparameters, and checkpoint are frozen, run
the corresponding test evaluation once by replacing `val.jsonl` with
`test.jsonl` and `val_metrics.json` with `test_metrics.json`. Eval always
checks that training and evaluation `game_id` sets are disjoint; there is no
disable switch or legacy dataset-path fallback.

Ordinary tests are self-contained and do not require formal data:

```bash
python -m pytest -q
```

Formal-data smoke tests are explicit and read-only:

```bash
RUN_TWD_TOM_REAL_DATA_TESTS=1 python -m pytest -q tests/twd_tom
```

Do not commit `data/`, `datasets/`, `outputs/`, or model checkpoints.

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

The model input is the structured `public_events` projection. The default
backbone is a randomly initialized Hugging Face `Qwen2Model` receiving
`inputs_embeds`; the training CLI also retains a direct stack of Hugging Face
`GPT2Block` layers. Neither option loads pretrained weights or uses a
tokenizer. The
first-order readout adds the last non-padding event state to each observer
embedding; rows also add one linear projection of that observer's private
knowledge. The second-order readout instead conditions every event token on
each modeled observer. For canonical zero-based observer seat `i` and
referenced player seat `j`, the shared relation index is `(j - i) mod 7`; a
separate index 7 is used when that token field has no player. Existing
structured fields are routed exactly as encoded: `turn_start` and
`public_speech` speakers, `speech_action` subjects and objects, vote voters and
targets, and exile/death objects. Three shared embedding tables cover speaker,
subject, and object/target relations, with shared self-relation flags. The
resulting relative public states have shape `[B,7,L,256]`; batch and observer
dimensions are flattened to `[B*7,L,256]`, and shared observer queries of shape
`[B*7,1,256]` use one shared attention module before residual LayerNorm. It
remains public-only. Both paths use the same shared 21-class output projection
and masked soft-target categorical cross entropy.

Only the second-order training Dataset applies deterministic cyclic player-ID
rotation. For epoch `e`, sample index `n`, and training seed `s`, its shift is
`(s + e + n) mod 7`; every structured player field and supervision mapping is
rotated consistently before rebuilding the canonical pair target. Seven
consecutive epochs cover all rotations for every sample. First-order data,
validation, evaluation, and shadow inference retain their canonical IDs. New
second-order checkpoints declare `observer_readout` as
`public_event_query_attention_v1` and `train_player_augmentation` as
`cyclic_rotation_v1`. They additionally declare `observer_event_conditioning`
as `cyclic_relative_player_relations_v1` and
`second_order_subject_supervision` as
`post_completed_public_speech_pre_next_action_v1`. Formal second-order samples
are limited to a complete `public_speech` immediately followed by the next
reasoning player's `turn_start`. All label-valid observer rows at that shared
cutoff are supervised except the reasoning player's own row. Vote, phase,
death, exile, and other system-event boundaries are excluded. New
checkpoints missing or mismatching that architecture and supervision contract
are rejected rather than converted. The first-order checkpoint contract is
unchanged.

For pair probabilities `q[i, omega]`, the sole player-level projection is

`m[i,j] = sum_{omega containing player j} q[i,omega]`.

The resulting marginal matrix has shape `[7,7]`; each value is the probability
that the corresponding player belongs to the two-Werewolf pair, so every row
sums to two. It is not a seven-class softmax, is not divided by two, and its
diagonal is not masked. A diagonal entry means that the modeled observer's
predicted pair belief includes that observer as a Werewolf; it is not a
separate notion of self-suspicion. The joint pair distribution cannot in
general be recovered from these marginals. Three-way decision remains outside
this contract.

## Online second-order shadow inference

`run_random.py` can load one explicit second-order checkpoint and write an
independent JSONL prediction log. The three shadow arguments—checkpoint,
device, and output path—must be supplied together. The output path must be new
and its parent directory must already exist. The optional `--random_seed`
exposes the runtime's existing deterministic role/profile assignment control;
shadow mode never changes it implicitly.

At each `speech` or `speech_pk` turn, inference runs after the matching
`turn_start` has entered `public_events` and before the acting agent generates
its speech. The model receives only the canonical structured public-event
prefix. It produces a `[7,21]` pair-probability matrix whose rows sum to one;
the fixed incidence projection above produces a `[7,7]` wolf-marginal matrix
whose rows sum to two. Both are logged without logits, roles, private
knowledge, or labels. No legacy `suspicion_matrix` alias is written. The result
is not added to observations, prompts, actions, votes, environment state, or
the original game log. Each record also contains `supervision_boundary` and
`observer_supervision_mask [7]`, calculated from the same completed-speech and
other-player contract used by formal supervision. Predictions remain present
for all seven observers; rows outside the formal supervision mask are logged
but are not described as supervised.
