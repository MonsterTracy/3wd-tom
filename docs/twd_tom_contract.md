# tom-v2 belief snapshot 契约

## 时间边界

每个 raw sample 在 `speech` 或 `speech_pk` 动作生成前创建。冻结的 `public_events` 必须截止于与当前 speaker 对应的 `turn_start`，且 `label_cutoff_step_idx == step_idx`。

末尾 `turn_start` 是证明“轮到谁发言”的采样边界标记，不是已经发生的公开语义。canonical artifact 必须保留它；Dataset 构造模型特征时必须且只能删除这一条末尾标记，使训练输入与未来对候选发言追加后的反事实输入采用同一个 completed-history 定义。

当前冻结版本：

- raw sample：`classic7_pre_speech_player_suspicion_v5`；
- label prompt：`classic7_pre_speech_player_suspicion_prompt_v6`；
- label provenance：`alive_observer_readonly_pre_speech_report_v3`；
- target conversion：`hard_knowledge_consistent_sparse_suspicion_uniform_support_v2`；
- target semantics：`relative_suspicion_matrix_v1`；
- public-only model input scope：`completed_structured_public_events_without_terminal_turn_start_v1`；
- private-conditioned model input scope：`completed_structured_public_events_plus_observer_hard_knowledge_v1`；
- public event：`classic7_public_event_sequence_v4`；
- speech annotation：`classic7_speech_annotation_v3`；
- speech action ontology：`classic7_speech_action_v1`；
- speech parser prompt：`classic7_speech_parser_v3`；
- public speech realization prompt：`classic7_public_speech_realization_prompt_v1`。

旧版本不做兼容读取或隐式迁移；需要重新采集。

## 观察者

采集对象是当前公开存活玩家。对每个 `observer_id`：

1. Environment 提供该玩家在当前边界合法可见的 observation；
2. Environment 提供该玩家的确定性 hard knowledge；
3. 调用该 playing agent 的 `report_suspected_werewolves_readonly()`；
4. query 前后 agent-owned state 必须完全相同。

不得向 collector 注入全局真实角色、其他玩家私有信息或未来事件。

## 原始标签

成功报告的原始标签是按 canonical player 顺序保存的 `suspected_werewolves` 集合。它表示观察者在合法私有信息状态下的狼人怀疑支持集，而不是公开指控、真实角色、完整双狼人组合或 reporter 概率。

对观察者 `i`，令 `K+` 为 Environment 规则推导并闭包后的已知狼人，`K-` 为已知非狼人，`R = K+ - {i}`，`F = K- ∪ {i}`。成功集合 `S` 必须满足：

```text
R ⊆ S
S ∩ F = ∅
```

因此已知的其他狼人必须出现，观察者自身和已知非狼人不得出现；`R` 非空时空集合非法。每个请求使用 observer-specific JSON Schema：候选 enum 在调用前排除 `F`，`minItems=|R|`，`maxItems=|P-F|`。重复项由本地 parser 拒绝，而不把 vLLM xgrammar 不支持的 `uniqueItems`、`contains` 或 `minContains` 发给服务端。计数约束不能证明具体成员已经包含，因此去重与 `R ⊆ S` 仍由本地校验强制执行。

纯解析或语义失败时，系统保持同一个冻结 base prompt、observation 和 hard knowledge，只在下一次请求末尾附加上一次本地验证错误，并重新生成完整 JSON，最多 3 次；接受首个完全合法的原始集合。反馈不包含修复后的候选响应，也不改变 hard knowledge。每次尝试的原始响应、状态和错误写入 `call_audit.json`。这属于可审计的 rejection generation，不会修复响应、删除非法成员、强制补入 `R`、补猜或概率化。底层 `BackendError` 另有最多 3 次显式、计入预算的传输尝试；纯 backend 连续失败不会再启动语义重生成。

在正式 canonical collector 中，这一条件不是延迟到 artifact validation 或训练时才检查：按 observer 顺序遇到首个三次生成后仍非 `ok` 的报告，不再请求其余 observer，当前局立即失败。失败 snapshot 不写入 raw JSONL；该局目录及逐次响应所在的 `call_audit.json` 移入 `failures/`，并保存带 digest 的 `failure.json`。collector 随后继续下一个预声明种子；这不是动态 replacement seed，失败种子也不会被重跑。

pilot 模式下，同样的 label failure 会跳过整条 PRE snapshot，而不是保存部分 observer 或伪造失败行；当前 speaker 随后直接提交固定 `no_commitment` 发言使游戏继续。per-game summary 记录缺失 PRE step，call audit 记录 label failure 与 fallback。pilot 始终 `canonical_eligible=false`，其不完整标签只能用于诊断，不能训练或物化。

raw sample 保存 public history 的两个 digest、观察者集合、每个观察者的 self-report、hard knowledge、状态与 backend provenance。它不保存 raw model response、私有 observation、pair target 或概率矩阵。

## Dataset 契约

tom-v2 只有一个 Dataset。输入是同一时间边界的结构化公开历史和存活观察者集合；raw label 直接读取 playing-agent self-report 中的 `suspected_werewolves`，不经过 external annotation、public-only reporter、ToM1/ToM2 materialization 或旧 lineage adapter。

Dataset 将每个观察者的合法非空怀疑集合确定性转换为长度 7 的稀疏 belief row：只在被列入的玩家上均分概率，其他玩家为 0。历史实验契约明确命名为 `legacy_v1`，其中空集合仍按旧实现插补为 hard-admissible non-self uniform；新契约明确命名为 `v1_empty_unobserved`，空集合保留在 canonical raw record 中，但派生监督标为 `label_observed=False`、target row 全零且不进入 CE/KL。二者具有不同的 target conversion、label-observation contract 和 checkpoint provenance，不得都简称为 V1。观察者自身对角线恒为 0。完整 target 为固定 `7×7`，行是 observer，列是 target player。

该矩阵的冻结语义是“相对怀疑质量”，不是每个玩家为狼的独立边缘概率，也不是两狼组合上的联合概率。softmax 只在每一行的非自身候选间分配单位质量；因此不能把 `B[i,j]=0.5` 解释为经过校准的 50% 狼人概率。checkpoint 以独立的 `target_semantics` 字段锁定这一解释，旧 checkpoint 不得隐式兼容。

Dataset 输出：

- `belief_targets`：`[7, 7]` 浮点张量；
- `observer_alive_mask`：`[7]` 布尔张量，只表示该时间点公开存活的 observer；
- `diagonal_target_mask`：`[7, 7]` 布尔张量，只排除 observer 自身列。

死亡玩家仍是合法 target，因此不得根据存活状态屏蔽 target 列。public-only 模式下，raw sample 中的 hard knowledge 只用于 label 合法性，不进入模型特征。private-conditioned 模式使用同一份 raw sample，额外输出 `known_werewolf_mask[7,7]` 与 `known_non_werewolf_mask[7,7]`；每行只编码该 observer 在当前 PRE 边界已经拥有并完成规则闭包的 `K+ / K-`。它不加入 `self_role`、全局角色真值、其他 observer 的 observation、未来信息或重构标签。两个 knowledge mask 必须成对出现、互斥，并与公开事件及 target 一起执行同一个座位循环旋转。`observer_supervision_mask = observer_scope_mask & label_observed_mask`；训练 Dataset 对非 `ok` 的存活 observer 直接失败，不补猜。

## Speaker cognition 同源契约

collector 返回 sample 后，从该 sample 中提取当前 speaker 的同一行成功报告，并将其封装为不可变 `SpeakerPreSpeechBelief`。对象同时锁定 observer、集合、hard knowledge、schema/prompt/provenance、step 和 structured-input digest；随后只能通过 `act_with_pre_speech_belief()` 进入紧邻的 strict day cognition。day cognition 不得重新生成、增删或替换这份私有怀疑支持，但公开 communication 仍可出于阵营策略与内部 belief 不同。

speaker report 非 `ok`、边界不匹配、agent 不支持专用入口或非 strict gameplay 时，必须在公开发言生成前失败。非 speaker 的报告不进入其行动上下文。

day cognition 先冻结与 PRE belief 同源的公开表达 intent，再通过独立的自然语言 realization 调用生成 1–4 句中文为主的公开原文；自然出现的常见英文词允许保留。该原文进入 immutable `public_speech`；随后 speech parser 只接收公开原文、speaker、day 和 phase，解析到独立的 `speech_annotations.jsonl`。生成器 intent 不能直接成为 canonical annotation 或模型输入。所有身份都禁止 cognition 候选生成 `point_as_werewolf(observer)`；公开原文若明确自称狼人，parser 仍应忠实记录，因为 parser 不做真值过滤。

strict realization 会拒绝空白/截断、非法 `playerN`、结构或控制文本泄漏、遗漏冻结具体目标，以及显式把当前 phase 说成其他天数的原文；普通英文不单独构成失败。realization 最多完整生成三次，后一次只追加上一次本地验证错误并保持同一冻结 intent，不修补或部分接收旧原文。speech parser 对同一原文最多生成三份完整响应，逐次保存原始响应与错误；不删行、不部分接受、不猜测目标。canonical 三次失败则拒绝当前局；pilot 保留 `status=error` 后继续，但永不可物化。

模型侧使用 14 类冻结 ontology，其中 `point_as_non_werewolf` 表示泛化的非狼/好人判断，`point_as_villager` 只表示具体村民身份判断，查验非狼使用 `check_as_non_werewolf`。`abstain_intent` 与 `no_commitment` 使用空 target。完整事件、annotation、失败和重解析契约见 `public_speech_event_contract.md`。

## Canonical materialization

`script/twd_tom/materialize_canonical_belief_dataset.py` 只接受 `collection_mode=canonical`、成功局数量达到冻结 target、所有成功局 fallback/label snapshot failure/speech annotation error count 为 0、每个 PRE boundary 都有完整成功 snapshot，且具有合法 `plan.json`、成功 `summary.json`、完整 per-game/per-failure digest、seed outcome 分区和 snapshot SHA256 链，并不存在 batch-level `batch_failure.json` 的 canonical batch。预声明池中的失败局保留在 `failures/` 供审计，但永不进入物化。`--mode pilot` 产生的批次即使没有实际 fallback 也不能进入物化。随后按稳定 SHA256 排名将成功 game 分配到 train、validation、test。一个 game 的所有 PRE snapshots 必须进入同一 split；输出记录保留原始 provenance 和 PRE boundary metadata，不执行 snapshot-level split。

物化前运行 `script/twd_tom/audit_canonical_belief_data.py`。该入口先验证上述成功批次摘要链，再验证全部 raw labels 可进入唯一 Dataset，并报告 suspicion support、原始/保留 token 长度以及在训练 `max_seq_len` 下的截断数量；audit 不修改原始记录。物化目录原子生成 `train.jsonl`、`validation.jsonl`、`test.jsonl` 和 `split_manifest.json`；manifest 绑定来源批次摘要、game-level split、输出文件 SHA256 与自身 digest。

`split_manifest.json` 是训练与评估的强制 lineage，而不是旁路记录。训练入口只接受同一目录、同一 manifest 指定的 train 与 validation 文件；评估入口只接受该 manifest 指定的 test 文件，并要求 checkpoint 记录相同的 manifest digest。manifest 校验同时检查三份输出的 SHA256、行数、game_id 集合和三路不相交；test 必须与 train、validation 两者都不重叠。

## 唯一来源

`label_source = playing_agent_readonly_self_report`。public-only reporter、external offline annotation 和 PBM 不是合法训练标签来源。

## 模型与目标函数

模型有两条显式且互不兼容的 checkpoint 契约。public-only 路径只输入由 public event 与对应 speech annotation 投影出的已经完成的结构化 `public_history < t`；private-conditioned 路径在完全相同的公开输入上，为每个 observer query 加入其自身 `K+ / K-` 的相对座位 embedding 之和。两条路径都不编码自然语言 `raw_text`，共享七个 canonical observer query 参数，且 raw snapshot 的末尾 `turn_start` 不进入特征。私有 embedding 只改变 query representation；它不修改 target、不屏蔽 logits、不重写模型响应，也不增加规则型输出兜底。输出始终为 `belief_logits[B, 7, 7]`，直接对应同一个 `belief_targets[B, 7, 7]`。

对每个存活 observer，在六个非自身 target 位置上计算 soft-target cross entropy；其中 hard knowledge 不允许的列已有 target 概率为 0。batch loss 是所有存活 observer 行的算术平均。死亡 observer 行不参与监督，死亡 player 列仍参与其他 observer 的分布。训练和 checkpoint 选择只使用这一 CE 目标；评估额外报告 target entropy 以及 `KL=CE-H(target)`，用于把标签本身的不确定性与模型相对怀疑质量区分开。不存在第二个 KL loss。checkpoint、metrics、train 和 eval 不保存旧任务阶数或组合类别字段。

评估中的 top-1 指标定义为预测最高概率集合是否与 soft target 的正概率支持集相交，不再把 `target.argmax()` 当作唯一类别。两条路径都报告六个非自身玩家均匀分布的固定 baseline。private-conditioned 路径还报告 `private_admissible_uniform`：对每个 observer 排除自身与 `K-` 后在剩余 target 上均分。私有模型的主 reducible-gap 指标必须相对这一更强 baseline 计算，不能把硬知识已经排除的候选当成模型学到的收益；CE loss 本身仍在全部六个非自身位置上计算。

`PrefixBeliefPredictor` 与 Dataset 共用同一个 strict-PRE 截断。public-only checkpoint 每次按完整公开历史重算并输出完整 `belief_matrix[7,7]`。private-conditioned checkpoint 只允许 `predict_observer()`：调用方必须显式传入当前 actor 自己的闭包后 `K+ / K-`，接口只返回该 observer 的长度 7 向量，避免 gameplay 调用方意外把其他角色私有信息拼入当前 actor 视角。两者都不缓存或递推 belief state，也不把预测解释为校准的独立狼人概率。

该 belief predictor 是 ONUW 式功能性 ToM 的状态预测基础，不等于完整行动闭环。完整闭环还需要在 gameplay 中对候选公开发言逐一追加、重新预测矩阵，并用当前玩家在其他观察者行中的相对怀疑变化进行候选评分；该决策层在实现并通过审计前不得宣称已经完成。
