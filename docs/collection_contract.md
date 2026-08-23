# tom-v2 Phase 1 采集冻结说明

当前采集契约冻结为 playing-agent readonly self-report：

- 边界：公开 `speech` / `speech_pk` 生成前；
- 观察者：当前公开存活玩家；
- 信息：各观察者当时合法可见的私有 observation 与确定性 hard knowledge；
- 标签：满足 hard knowledge 约束的 `suspected_werewolves` 符号集合；
- 副作用：query 前后 playing agent 状态必须相同；
- 同源行动：当前 speaker 的同一份成功 PRE report 以冻结输入进入随后 strict day cognition；
- 禁止：真实角色注入、未来信息、external reporter、public-only reporter、PBM、概率标签、repair、retry 或 fallback；任何身份都不得生成指向自己的 `point_as_werewolf`。

正式 canonical batch 还必须满足：

- YAML `pipeline` 的 schema/projection 版本与当前代码常量完全一致；
- CLI 的连续 seed 范围与 game count 和 `pipeline.collection` 完全一致；
- 每个实际 backend dispatch 都计入 gameplay 或 belief 调用预算，speech parser 等未显式包裹的调用计入 gameplay；
- 达到 category/total call 上限后，在下一次请求发送前失败；每次请求前后检查单局 wall-clock 上限；
- 每局无论成功或失败都保存 `call_audit.json`，成功 summary 同时汇总调用数；
- 任一存活 observer 的报告不是 `status=ok`，该局在 artifact validation 阶段失败，不能成为 canonical completed game。

采集完成后必须运行：

```bash
python -m script.twd_tom.audit_canonical_belief_data \
  --canonical-root /data/yuxiao/3wd-tom/canonical_data/<run_id> \
  --max-seq-len 256 \
  --output /data/yuxiao/3wd-tom/canonical_data/<run_id>/belief_data_audit.json
```

审计会再次验证 raw schema、prompt/provenance、game/step 唯一性、全部 observer 成功状态与 Dataset 可加载性，并报告 label support size 和结构化序列截断率。只有 `status=PASS` 的数据才进入 game-level materialization。

历史 V2.7/ToM1/ToM2 设计以 Git 历史为准，不再作为 tom-v2 active contract。
