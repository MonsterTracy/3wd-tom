# tom-v2 旧主线清理结论

本轮已经从 active dependency graph 移除以下旧研究路径：

- tom-v1 formal reporter、collector、pilot model 与 archive scripts；
- public-only belief snapshot collector 与 reporter；
- Public Belief Matrix 的 collection/model/train scripts；
- canonical trajectory 到 external offline annotation C1 的标签路径；
- D ToM1/ToM2 materializer 与 game split；
- online ToM2 shadow inference；
- Dataset/train/eval 中对应的旧 lineage adapter、manifest 和 compatibility branch。

保留的唯一 label pipeline 是：公开发言前冻结边界，读取目标 playing agent 的合法私有 observation，执行 readonly self-report，以 hard knowledge 拒绝不可能的支持集，保存 `suspected_werewolves` 符号集合，再由唯一 Dataset 确定性转换为 7×7 observer-conditioned belief target。speaker 的同一份成功报告同时作为随后 strict day cognition 的冻结输入。

Dataset、模型、loss、metrics、训练与 evaluation 均不再包含 `tom_order`、ToM1/ToM2、pair target、21 类投影、pair KL/accuracy 或私有知识模型输入。模型直接预测 `belief_logits[B, 7, 7]`，并以存活 observer 与非对角 target mask 计算分布损失。
