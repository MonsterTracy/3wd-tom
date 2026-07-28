class TWDMStrategy:
    def __init__(self, config=None):
        self.config = config or {}

    def build_hint(self, observation):
        if not self.config.get("enable_strategy", True):
            return ""

        identity = observation.get("identity", "")
        phase = observation.get("phase", "")

        role_hint = self._build_role_hint(identity)
        phase_hint = self._build_phase_hint(identity, phase)

        if not role_hint and not phase_hint:
            return ""

        return "TWDM策略提示：" + role_hint + phase_hint

    def _build_role_hint(self, identity):
        if identity == "Werewolf":
            return (
                "你当前是狼人阵营玩家。你的核心目标是在隐藏真实身份的前提下帮助狼人阵营获胜。"
                "私有信息只能用于内部策略判断，公开发言必须伪装成普通好人公开视角。"
                "禁止暴露狼人队友身份、夜晚刀人目标、狼队讨论内容，以及谁被刀但没死。"
                "禁止声称某人是被女巫救下的人，不能把此类私有判断说成确定结论。"
                "公开理由必须来自白天发言、公开死亡结果、公开投票结果、公开身份声明和公开逻辑矛盾。"
                "不能把私有信息包装成确定公共事实；如需利用夜晚信息，只能转化成公开视角下的模糊怀疑。"
                "尽量制造好人阵营之间的怀疑，推动白天投出非狼人玩家。"
            )

        if identity in ("Villager", "Seer", "Witch", "Guard"):
            return (
                "你当前是村民阵营玩家。你的核心目标是通过发言、投票和技能信息找出狼人。"
                "请保持信息一致性，避免无根据地误导好人阵营。"
            )

        return ""

    def _build_phase_hint(self, identity, phase):
        if identity == "Werewolf":
            if "skill_wolf" in phase:
                return " 当前是狼人夜晚行动阶段，请优先选择可能是神职、发言强势或对狼人威胁较大的目标。"
            if "speech" in phase:
                return (
                    " 当前是白天发言阶段，请保持普通好人的公开视角。"
                    "例如，平安夜只能说明可能存在女巫救人或其他情况，不能确定某人就是被女巫救下的好人。"
                )
            if "vote" in phase:
                return " 当前是投票阶段，请选择有利于狼人阵营的出局目标，优先推动非狼人玩家出局。"

        if identity in ("Villager", "Seer", "Witch", "Guard"):
            if "speech" in phase:
                return " 当前是白天发言阶段，请基于公开信息和你的身份信息进行推理，尽量找出狼人。"
            if "vote" in phase:
                return " 当前是投票阶段，请优先投给你认为最可能是狼人的玩家。"
            if "skill_seer" in phase:
                return " 当前是预言家查验阶段，请优先查验发言可疑或影响局势较大的玩家。"
            if "skill_witch" in phase:
                return " 当前是女巫行动阶段，请谨慎使用解药和毒药，避免误伤好人。"
            if "skill_guard" in phase:
                return " 当前是守卫行动阶段，请优先保护关键好人或可能遭到狼人攻击的玩家。"

        return ""
