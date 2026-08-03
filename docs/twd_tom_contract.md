# Classic7 actor-perspective ToM data contract

## Research perspective

Three perspectives must not be conflated:

- A: a public-only belief attached to a public history, without a designated
  acting player's private information.
- B: the current action role's perspective. At a decision point, the actor
  `o` reasons from public history plus only `o`'s legal private knowledge.
- C: a belief-owner-indexed collection in which each player's private state is
  treated as a separate model input.

The formal research definition selects **B**. `reasoning_player` is the player
currently making the decision and always equals `current_speaker` at the
pre-speech snapshot. `belief_owner` is the player whose internal world belief
is described by one target row. These are different roles in ToM2.

The reasoning input is exactly:

`I_o = public_history + reasoning_player_id + o's legal private knowledge`

It excludes every other player's identity, hard knowledge, private reporter
payload, agent memory, target, and target provenance. It also excludes the
unfinished current action and all future events.

## World space and targets

There are seven fixed seats and exactly two Werewolves. The sole world space
is the lexicographic ordering returned by `canonical_wolf_pairs()`:

`(player1,player2), (player1,player3), ... , (player6,player7)`

It always contains 21 classes.

- ToM1 is `self_world_belief = b_o(omega)`, shape `[21]`.
- ToM2 is `other_world_beliefs = q_(o->i)(omega)` for the six `i != o`,
  shape `[6,21]`.
- `other_player_ids`, shape `[6]`, records the canonical row-to-seat mapping.
- `other_target_mask`, shape `[6]`, marks which direct belief-owner reports
  succeeded.

The contract is not `[7,15]`: every row uses the same global 21-class
ordering. Hard knowledge changes a row's `legal_world_mask`, not its class
vocabulary. A non-Werewolf's self belief normally assigns zero probability
to every pair containing self but remains 21-dimensional. A Werewolf who
legally knows the team assigns support to the true pair containing self; this
expresses identity knowledge, not self-suspicion.

A ToM2 row is the belief owner's direct self report used as the target for
what `o` believes about that owner. There is no blanket rule excluding pairs
that contain `o`, because an owner may falsely believe that `o` is a wolf.
There is also no blanket rule excluding pairs that contain the belief owner,
because a Werewolf owner legally knows a pair containing self.

## Pre-speech boundary

Collection occurs after the environment appends the current actor's
`turn_start` and before that actor generates a public speech or PK speech.
The frozen `public_events` prefix must end in that matching `turn_start`.
The current action appears only in a later snapshot.

Each raw record stores `event_idx`, `day`, `phase`, `current_speaker`, the
public-history cutoff, `current_action_used=false`,
`future_information_used=false`, and the public-event digest. Only players
alive at this boundary are asked to report. Dead players retain canonical
rows with a missing target and are not sent reporter requests.

## Direct pair-belief self report

Formal collection calls `ReadonlyPairBeliefSelfReporter`. A belief owner
directly returns exactly:

```json
{"pair_probabilities":[0.0, 0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.1]}
```

The array is not sorted, projected, normalized, completed, or reconstructed
from player marginals. It must contain exactly 21 finite non-negative
numbers and sum to one under the repository's established
`rtol=1e-5, atol=1e-6` probability tolerance.

The environment remains the sole source of `known_werewolves` and
`known_non_werewolves`. A pair is legal exactly when it contains every known
Werewolf and intersects no known non-Werewolf. Probability above the same
formal tolerance on an illegal pair is a `semantic_error`. The raw reporter
output and parsed output remain available for audit, while the usable
`pair_probabilities` target stays null. Parse, semantic, or reporter failures
never become `ok` and are never retried, normalized, carried forward, or
expert-completed.

## Canonical reporter payload

`LLMAgent.build_readonly_pair_belief_payload()` is the sole payload builder.
It creates a JSON-safe object containing the exact messages and request
parameters subsequently passed unchanged to
`report_pair_belief_self_readonly()`.

The payload includes only the belief owner's legal observation-derived
context: self role, filtered private role history, `notes`/`vote_reason`
memory when present, the frozen public history, and environment-generated
hard knowledge. It contains no API key, authentication header, endpoint, or
absolute filesystem path. The complete payload is serialized in the raw
record and hashed as canonical JSON. The complete prompt is inside that
payload and receives its own SHA-256. Saving and sending never rebuild
separate approximations.

## Raw schema and provenance

The sole formal raw schema is:

`classic7_fixed_two_wolves_actor_perspective_direct_pair_belief_self_reports_v1`

Its annotation version is:

`classic7_direct_pair_belief_self_reports_v1`

Each record contains one `reasoning_player_id=current_speaker`, the canonical
reasoning input and digest, and seven canonical `player_reports`. Each player
row records alive state, status/error, target or null, hard knowledge,
reporter payload and digest, raw and parsed output, hard-knowledge validation,
backend alias, exact resolved model name, prompt/parser versions and hashes,
sampling parameters, optional reporter seed, and report provenance.

Collection provenance additionally records generator name/version, Git
commit, clean/dirty worktree state, UTC timestamp, game seed,
repository-relative source config and SHA-256, resolved runtime-config hash,
per-backend resolved-config hashes, public-event digest, reporter payload
digests, prompt digests, and resolved reporter routes. Personal absolute paths
and credentials are not serialized.

## Actor-perspective offline mapping

`build_actor_perspective_sample()` maps one raw snapshot to exactly one sample.
It never expands a snapshot into seven perspectives.

- `self_pair_target` is the current speaker's successful direct report or
  null.
- `other_player_ids` is canonical player order with the current speaker
  removed.
- `other_pair_targets` has six 21-class rows in exactly that order.
- `other_target_mask` is false for dead, missing, or failed reports. A masked
  row uses a zero storage value but is never treated as a valid target.
- the reasoning input contains only public history, the reasoning player ID,
  and that player's legal private knowledge.

Other players' hard knowledge and reporter payloads exist only in raw
label/provenance rows and never enter `reasoning_input`.

## Historical data

Existing `data/qwen25/raw_tom2.jsonl` and the current `tom1/` and `tom2/`
splits use older provenance-incomplete contracts. They are immutable
historical diagnostic data and must not be mixed with the new actor-
perspective schema for training.

The historical `suspected_werewolves` validator and
`project_suspicion_to_pairs.py` remain only for reading and diagnosing those
existing records. They are not called by the formal collector. The new formal
path contains no hard-knowledge-only, exact-private-pair, carry-forward, or
other expert-completion mechanism.

## Current stage boundary

This phase establishes collection, raw schema, provenance, and an offline
mapping contract only. It does not implement a new `[6,21]` model head,
Dataset loader, loss, metric, checkpoint, training run, evaluation run, or
shadow behavior. Existing model numerical paths remain frozen and do not yet
represent this new formal research target.

Before any later model work, a small separately approved smoke collection in
the new schema must be audited for payload equality, support validity,
missing-report behavior, provenance completeness, and the pre-speech causal
boundary.
