# tom-v2 Phase 1 采集冻结说明

当前采集契约冻结为 playing-agent readonly self-report：

- 边界：公开 `speech` / `speech_pk` 生成前；
- 观察者：当前公开存活玩家；
- 信息：各观察者当时合法可见的私有 observation 与确定性 hard knowledge；
- 标签：`suspected_werewolves` 符号集合；
- 副作用：query 前后 playing agent 状态必须相同；
- 禁止：真实角色注入、未来信息、external reporter、public-only reporter、PBM、概率标签、repair、retry 或 fallback。

历史 V2.7/ToM1/ToM2 设计以 Git 历史为准，不再作为 tom-v2 active contract。
