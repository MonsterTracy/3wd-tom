# tom-v2 Phase 1 采集冻结说明

当前采集契约冻结为 playing-agent readonly self-report：

- 规则基线：七人预言家—女巫局（2 狼人、1 预言家、1 女巫、3 村民）；无存活狼人时好人胜，存活狼人数达到或超过存活非狼人数时狼人胜；

- 边界：公开 `speech` / `speech_pk` 生成前；
- 观察者：当前公开存活玩家；
- 信息：各观察者当时合法可见的私有 observation 与确定性 hard knowledge；
- 标签：满足 hard knowledge 约束的 `suspected_werewolves` 符号集合；
- 副作用：query 前后 playing agent 状态必须相同；
- 同源行动：当前 speaker 的同一份成功 PRE report 以冻结输入进入随后 strict day cognition；
- 公开发言：冻结 intent 后单独生成自然中文，再由只看公开原文的 parser 生成 `speech_annotations.jsonl`；
- 禁止：真实角色注入、未来信息、external reporter、public-only reporter、PBM、概率标签，以及 label 语义 repair、删除非法成员、fallback label、补猜；任何身份都不得生成指向自己的 `point_as_werewolf`。

## 有界重试与运行模式

- backend 客户端的 SDK retry 固定为 0；预算代理只对 `BackendError` 做最多 3 次显式尝试。每个实际 dispatch 都计入调用预算，失败尝试和下一次尝试写入 `call_audit.json`。
- strict gameplay 的 belief/day cognition、vote、night action 和 speech realization 若发生截断、结构解析、非法动作或发言质量错误，只重生成当前阶段，最多 3 次。不会因为一个阶段失败而重跑整局；纯 backend 连续失败在第 3 次 dispatch 后直接抛出，不会再进入语义重生成。若每次逻辑生成都先经历瞬时 backend 失败、随后成功返回但语义仍非法，单阶段理论上最多可产生 9 次实际 dispatch，全部受 call budget 约束并进入审计。
- PRE readonly label self-report 使用 observer-specific JSON Schema：候选 enum 在请求前排除 observer 自身和 hard knowledge 中的已知非狼人。重复项由本地 parser 拒绝，因为 vLLM 的 xgrammar 明确不支持 `uniqueItems`；纯解析或语义失败时重新执行同一个 readonly label query，最多 3 次。每次原始响应、状态和错误写入 `call_audit.json`。这是有界 rejection generation，不会修改返回集合、删除非法成员、补入 hard knowledge 或制造替代标签。
- 独立 speech parser 对同一公开原文最多做 3 次完整生成；后一次只获得前一次的严格验证错误。每次原始响应与错误保存在 annotation v3；只接受首个整份合法响应，不删除非法行、不部分接受、不猜测目标。
- `--mode pilot` 允许 gameplay 三次生成耗尽后提交确定性的合法 fallback；若一个 PRE snapshot 的任一 observer 在 3 次 label 生成后仍失败，则整条 PRE snapshot 不写入 raw JSONL，当前 speaker 直接提交固定的 `no_commitment` 发言使游戏继续。speech parser 三次失败则保留 `status=error` 及全部 attempts，不伪造 action，并继续完成后续游戏与局数。整批始终为 `canonical_eligible=false`。
- `--mode canonical` 不允许任何 fallback。3 次 label 生成仍失败时终止；speech parser 三次失败会完整保存 attempts，并在该局 artifact 验收边界 fail closed。canonical validator/materializer 同时拒绝 pilot 批次、PRE 覆盖不完整、label snapshot failure、speech annotation error 或 gameplay fallback 非零的游戏。

正式 canonical batch 还必须满足：

- YAML `pipeline` 的 schema/projection 版本与当前代码常量完全一致；
- CLI 的连续 seed 范围与 game count 和 `pipeline.collection` 完全一致；
- 每个实际 backend dispatch 都计入 gameplay 或 belief 调用预算，speech parser 等未显式包裹的调用计入 gameplay；
- 每次 strict 公开发言通常包含 day cognition、自然语言 realization 和 1–3 次 speech parsing dispatch；
- 达到 category/total call 上限后，在下一次请求发送前失败；每次请求前后检查单局 wall-clock 上限；
- 每局无论成功或失败都保存 `call_audit.json`，成功 summary 同时汇总调用数；
- canonical 模式下，任一存活 observer 在 3 次生成后仍不是 `status=ok`，立即终止该局且不再请求其余 observer；失败 snapshot 不写入 raw JSONL，也不能成为 canonical completed game。pilot 模式遵循上面的“跳过整条 PRE snapshot + 固定无承诺发言”诊断策略。
- 每个 `public_speech` 必须在 `speech_annotations.jsonl` 中有且只有一条由独立 `llm_parser` 产生、按 event/speaker/raw-text digest 绑定的 annotation；canonical 拒绝 `status=error`，pilot 保留错误但不获得 canonical 资格。
- 每个 belief snapshot 内嵌的 `speech_annotations` 必须与同局 canonical `speech_annotations.jsonl` 在该 snapshot 公开事件截止点上的规范化前缀完全相同；不能只依赖 snapshot 自身 digest。
- 每个 completed game 必须使用记录的角色、environment seed 和 submitted actions 通过 deterministic environment replay；重放不得重新调用 LLM 或 speech parser。

每局成功 artifact 至少包含 `trajectory.json`、`observer_views.json`、`belief_snapshots.jsonl`、`speech_annotations.jsonl`、`call_audit.json` 和 `summary.json`。其中 trajectory/public events 保存公开原文，annotation sidecar 保存版本化结构语义；snapshot 只保存截至自身 PRE 边界的 sidecar 前缀。重放只验证 simulator 与原文事件，不重新生成或重新解析发言。

每个 completed game 也可以独立执行重放审计：

```bash
python -m script.twd_tom.replay_canonical_trajectory \
  /data/yuxiao/3wd-tom/canonical_data/<run_id>/games/<game>/trajectory.json \
  --observer-views /data/yuxiao/3wd-tom/canonical_data/<run_id>/games/<game>/observer_views.json
```

该命令只执行已记录的 submitted actions，并逐步核对 observation、public events、alive/terminal、PRE/POST observer views、winner 和 artifact digests，不调用模型或 speech parser。

采集完成后必须运行：

```bash
python -m script.twd_tom.audit_canonical_belief_data \
  --canonical-root /data/yuxiao/3wd-tom/canonical_data/<run_id> \
  --max-seq-len 256 \
  --output /data/yuxiao/3wd-tom/canonical_data/<run_id>/belief_data_audit.json
```

审计会先拒绝含 `batch_failure.json`、缺少成功摘要或 plan/batch/per-game digest 与 snapshot SHA256 不一致的批次，再验证 raw schema、prompt/provenance、game/step 唯一性、全部 observer 成功状态与 Dataset 可加载性，并报告 label support size 和结构化序列截断率。只有 `status=PASS` 的数据才进入 game-level materialization。

历史 V2.7/ToM1/ToM2 设计以 Git 历史为准，不再作为 tom-v2 active contract。
