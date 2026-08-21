# Architecture

This document maps the current implementation without introducing components
that are not present in the repository. Collection V2.7 runtime paths are
frozen; the diagrams describe them but do not redefine their behavior.

## Gameplay flow

```mermaid
flowchart LR
    E["WerewolfTextEnvV0"] --> O["Player observation"]
    O --> A["Configured agent"]
    A --> C["Canonical valid action"]
    C --> S["env.step(action)"]
    S --> E
    S --> GL["Game log"]
    S --> PE["Append-only public_events"]
```

`werewolf/envs/werewolf_text_env_v0.py` owns phases, legal actions, state
transitions, observations, game logs, and public-event publication. Agent
construction and backend routing are performed by `run_battle.py` or
`run_random.py`; the environment does not resolve API credentials or model
names.

## Public speech flow

```mermaid
flowchart LR
    PC["Legal private context"] --> B["Transient belief"]
    PS["Authoritative public state"] --> B
    B --> SP["Direct public speech"]
    SP --> P["SpeechPerceiver"]
    P --> SA["Structured sp_actions"]
    SP --> EV["public_speech event"]
    SA --> EV
```

The strict speech path is orchestrated by `GPTAgent`. A fresh transient belief
is generated from the legally filtered observation, then a second model call
produces natural-language public speech directly. Prompt construction lives in
`prompt_template_v0.py`. The online `SpeechPerceiver.parse()` remains tolerant
and runs only after speech generation, so parser failure does not stop a game;
strict parsing is reserved for offline audit tools.

## Canonical trajectory and supervision flow

```mermaid
flowchart LR
    G["Simulator"] --> A["A canonical trajectory"]
    G --> C0["C0 observer-view provenance"]
    A --> C1["Offline annotation C1"]
    C0 --> C1
    A --> D["Offline materialization D"]
    C0 --> D
    C1 --> D
    D --> S["Deterministic game-level split"]
    S --> DS["TWDToMDataset"]
```

`CanonicalGameInteractionTrajectoryRecorder` is the only current-mainline
gameplay collector. It records A transitions and C0 PRE/POST observer views;
it does not call a label reporter. `werewolf.offline_annotation` derives C1
private-conditioned and public-only suspicion annotations from frozen A/C0.
`werewolf.offline_materialization` validates those sources and emits D ToM1
and ToM2 records. Deterministic observer hard knowledge is owned by
`werewolf/observer_knowledge.py`. The reproducible filesystem entry point for
these offline stages is `script/twd_tom/materialize_canonical_dataset.py`.

The historical online belief collectors and the earlier formal-ToM pilot are
preserved under `archive/legacy_tom`; they are not sources for canonical D.

## Training flow

```mermaid
flowchart LR
    D["Validated D ToM1 / ToM2 JSONL"] --> ST["Game-level hash split"]
    ST --> DS["TWDToMDataset"]
    DS --> F["Structured event features"]
    F --> B["Selected ToM backbone"]
    B --> H["Shared 21-class head"]
    H --> L["Masked soft-target cross entropy"]
    L --> C["best.pt / last.pt"]
    C --> E["Explicit validation or test evaluation"]
```

`script/twd_tom/train.py` and `script/twd_tom/eval.py` are the formal model
entry points. Training and validation datasets are explicit and game-disjoint.
The default backbone is a randomly initialized `Qwen2Model` receiving
`inputs_embeds`; the CLI also retains a direct `GPT2Block` stack. Neither path
downloads pretrained weights or tokenizes raw speech.

## Source locations

| Concern | Location |
|---|---|
| Environment | `werewolf/envs/` |
| Agent and speech-plan contracts | `werewolf/agents/` |
| Backend abstraction | `werewolf/backends/` |
| Speech perception and belief reporting | `werewolf/speech/` |
| Canonical trajectory and offline labels | `werewolf/trajectory.py`, `werewolf/offline_annotation.py`, `werewolf/offline_materialization.py` |
| ToM schemas, Dataset, model, loss, metrics | `werewolf/models/twd_tom/` |
| Collection and processing CLIs | `script/twd_tom/` |
| Historical ToM implementations | `archive/legacy_tom/` |
| Deterministic regressions | `tests/` |
