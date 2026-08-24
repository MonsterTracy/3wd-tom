# 公共事件与发言语义冻结契约

本契约冻结 tom-v2 的第二部分：公开事件如何保存、自然语言发言如何结构化，以及哪些信息可以进入 belief backbone。目标是同时保留可审计的原始语义和稳定、无泄漏的模型输入。

## 1. 冻结版本

- public event：`classic7_public_event_sequence_v4`
- speech annotation：`classic7_speech_annotation_v2`
- speech action ontology：`classic7_speech_action_v1`
- speech parser prompt：`classic7_speech_parser_v2`
- PRE belief sample：`classic7_pre_speech_player_suspicion_v4`

版本不做隐式兼容或自动迁移。任一版本变化都必须产生新 canonical run；历史原文可以通过显式离线重标注生成新 annotation artifact，但不能原地覆盖旧 artifact。

## 2. Public event 是事实层

`public_events` 是按 `event_idx` 连续追加的公开事实序列。允许的事件和字段如下：

| event type | 固定字段 |
| --- | --- |
| `phase_change` | `event_idx`, `event_type`, `phase` |
| `turn_start` | `event_idx`, `event_type`, `speaker` |
| `public_speech` | `event_idx`, `event_type`, `speaker`, `raw_text` |
| `vote_result` | `event_idx`, `event_type`, `votes` |
| `exile_result` | `event_idx`, `event_type`, `exiled_players` |
| `death_announcement` | `event_idx`, `event_type`, `dead_players` |

`public_speech` 不保存 action、私有 cognition、角色真值、parser 输出或摘要。`raw_text` 是环境实际接受并向玩家公开的自然语言文本，是后续审计和重标注的唯一语义原件。结构化 ontology 无法表达的修辞、否定、条件、引用和语气仍保留在原文中，因此不会因为当前 action 集合有限而永久丢失。

## 3. Speech annotation 是可替换解释层

每局单独保存 `speech_annotations.jsonl`，每个 `public_speech` 恰好对应一行。固定字段为：

```text
schema_version, event_idx, speaker, raw_text_digest,
ontology_version, parser_prompt_version, parser_model_id, parser_call_id,
annotation_source, status, actions, raw_response,
error_type, error_message
```

绑定条件同时使用：

1. `event_idx` 指向唯一 `public_speech`；
2. `speaker` 与事件一致；
3. `raw_text_digest = SHA256(public_speech.raw_text)`；
4. annotation 按 `event_idx` 唯一升序，且完整覆盖所有公开发言。

`status=ok` 必须至少有一个 action；`status=no_action` 表示 parser 成功但没有可表示命题；`status=error` 必须没有 action，并保存明确错误。`raw_response` 保留 parser 的原始响应，便于复核，而不是把清洗后的结果冒充原始输出。

唯一合法的 annotation 来源是 `annotation_source=llm_parser`。环境只接受公开原文字符串，结构化动作必须由 speech parser 独立产生；parser `status=error` 的游戏不能进入正式 canonical 数据。

## 4. 冻结的 14 类 action ontology

每个 action 是 `[subject, action, object]`；subject 必须等于当前 speaker。目标型 action 的 object 是 `player1` 至 `player7`，无目标 action 的 object 是 `null`。

| action | 精确定义 |
| --- | --- |
| `point_as_werewolf` | 明确判断目标是狼人，且不是 speaker 自称查验所得 |
| `point_as_non_werewolf` | 明确判断目标非狼、是泛化好人或好人阵营，不具体化角色 |
| `point_as_villager` | 明确判断目标具体是村民/平民 |
| `point_as_seer` | 明确判断目标具体是预言家 |
| `point_as_witch` | 明确判断目标具体是女巫 |
| `support` | 明确支持、认可或站边目标及其观点 |
| `oppose` | 明确反对、不信任或质疑目标及其观点 |
| `check_as_non_werewolf` | speaker 明确声称查验目标得到非狼/好人结果 |
| `check_as_werewolf` | speaker 明确声称查验目标得到狼人结果 |
| `save` | speaker 明确公开声称救了目标 |
| `poison` | speaker 明确公开声称毒了目标 |
| `vote_intent` | speaker 明确表达本轮准备把票投给目标 |
| `abstain_intent` | speaker 明确表达本轮准备弃票，object 为 `null` |
| `no_commitment` | speaker 明确表示本轮暂不作正式表态，object 为 `null` |

采用 most-specific-source：一个具体命题不自动派生更泛化 action。比如“查验 player3 是好人”只产生 `check_as_non_werewolf`，不自动产生 `point_as_non_werewolf`、`point_as_villager` 或 `support`；“player3 可信”只产生 `support`。parser 只忠实抽取当前 speaker 的明确公开命题，不验证真假，不读取真实角色或私有技能记录。

## 5. 正式发言链路

```text
合法 PRE private belief
  -> strict day cognition 选择结构化 communication intent
  -> intent 冻结
  -> 独立 realization 调用生成 1–4 句自然中文
  -> raw_text 作为 public_speech 提交给环境
  -> 独立 parser 只读取 raw_text + speaker/day/phase
  -> speech_annotations.jsonl
```

realization prompt 可以得到冻结 intent 和此前公开历史用于自然衔接，但不能增加新正式命题。parser 不接收冻结 intent、private belief、真实身份或生成器隐藏 reasoning。即使 realization 与 parser 使用同一基础模型，它们也是不同请求、不同 prompt 和单向信息边界；canonical annotation 必须来自 parser 实际读到的公开原文。

## 6. 模型输入和 memory

当前 belief backbone 不直接编码 `raw_text`，只编码 public event 边界、系统事件和 annotation actions。原始文本仍完整归档，后续可以比较 parser 版本、扩展 ontology 或增加文本 encoder。

训练时的循环座位旋转只发生在 Dataset 的内存结构化视图中，旋转 observer、target、事件玩家引用和 annotation action；它不回写 canonical artifact，也不把旋转后的自然语言当作新原文或新标注来源。

不把 agent 内部隐藏 memory、私有 observation、chain-of-thought 或滚动摘要写入 public event。模型所需的公开 memory 由从游戏开始到时间边界 `t` 的完整 append-only public prefix 表示；截断由训练 `max_seq_len` 显式审计，不用不可审计摘要替代历史。

## 7. 复现与失败策略

“确定性复现”指使用记录的角色分配、environment seed 和 submitted actions，在不调用 LLM/parser 的条件下逐步复现 simulator 状态、公开事件、observer views、winner 与 digests。它不要求重新采样得到同一句 LLM 发言，也不在 replay 时重新解析原文。

正式采集遇到 realization 失败会在发言提交前停止；parser 失败可以让环境留下原文和错误 annotation，但该局随后必须在 canonical artifact validation 中失败并保留审计工件。禁止 retry、fallback action、从 generator intent 回填 annotation 或 replacement seed。

## 8. 验收条件

- `public_speech` 字段中不存在 `sp_actions` 或其他结构语义。
- 每条公开发言有且只有一个 digest 绑定的 annotation。
- ontology、parser prompt、raw sample 和 public event 版本写入配置及 artifact。
- PRE sample 的 structured-input digest 同时覆盖 public events 与 annotation prefix。
- Dataset 和 backbone 不读取 `raw_text`。
- canonical validator 拒绝缺失、重复、错 speaker、错原文 digest 和 parser error annotation。
- deterministic replay 不调用 agent、realization 或 speech parser。
