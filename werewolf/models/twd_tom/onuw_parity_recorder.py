"""In-memory strict-PRE recorder for new parity pilot games."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from werewolf.models.twd_tom.onuw_parity_dataset import (
    PARITY_GAME_SCHEMA_VERSION,
    bos_token,
    materialize_public_tokens,
    validate_parity_game,
)
from werewolf.models.twd_tom.onuw_parity_protocol import (
    CLASSIC7_ONUW_REFERENCE,
    CONTENT_PROFILES,
    MODALITY_PROFILES,
)
from werewolf.models.twd_tom.schema import PLAYER_NAMES, normalize_player
from werewolf.speech.onuw_role_guess_perceiver import (
    role_guess_reports_to_matrix,
)


class OnuwParityGameRecorder:
    """Accumulate one game; PRE is recorded before its speech is appended."""

    def __init__(self, *, game_id: str, content_profile: str, modality_profile: str):
        if not isinstance(game_id, str) or not game_id.strip():
            raise ValueError("game_id must be non-empty text")
        if content_profile not in CONTENT_PROFILES:
            raise ValueError("unsupported content_profile")
        if modality_profile not in MODALITY_PROFILES:
            raise ValueError("unsupported modality_profile")
        self.game_id = game_id
        self.content_profile = content_profile
        self.modality_profile = modality_profile
        self._tokens = [bos_token()]
        self._queries = []
        self._speech_action_counts = []
        self._label_audit = []
        self._speech_emotions = []

    def record_pre(
        self,
        *,
        step_idx: int,
        speaker_id: int | str,
        observer_ids: Sequence[int | str],
        role_guess_reports: Mapping[str, Mapping[str, Any]],
    ) -> None:
        if isinstance(step_idx, bool) or not isinstance(step_idx, int):
            raise TypeError("step_idx must be an integer")
        speaker = normalize_player(speaker_id)
        observers = [normalize_player(observer) for observer in observer_ids]
        if observers != [player for player in PLAYER_NAMES if player in observers]:
            raise ValueError("observer_ids must use unique canonical order")
        query_id = f"{self.game_id}:pre:{step_idx}"
        self._queries.append(
            {
                "query_id": query_id,
                "step_idx": step_idx,
                "speaker": speaker,
                "token_cutoff": len(self._tokens) - 1,
                "observer_ids": observers,
                "belief_target": role_guess_reports_to_matrix(
                    role_guess_reports, observer_ids=observers
                ),
            }
        )
        self._label_audit.append(
            {
                "query_id": query_id,
                "step_idx": step_idx,
                "speaker": speaker,
                "observer_ids": list(observers),
                "role_guess_reports": deepcopy(dict(role_guess_reports)),
            }
        )

    def sync_post_public_history(
        self,
        *,
        public_events,
        speech_annotations,
        speech_emotions,
    ) -> None:
        tokens, action_counts = materialize_public_tokens(
            public_events=public_events,
            speech_annotations=speech_annotations,
            speech_emotions=speech_emotions,
            content_profile=self.content_profile,
            modality_profile=self.modality_profile,
        )
        if tokens[: len(self._tokens)] != self._tokens:
            raise ValueError("public token history must be append-only")
        if action_counts[: len(self._speech_action_counts)] != (
            self._speech_action_counts
        ):
            raise ValueError("speech action counts must be append-only")
        self._tokens = tokens
        self._speech_action_counts = action_counts
        self._speech_emotions = deepcopy(list(speech_emotions))

    def finalize(self) -> dict[str, Any]:
        game = {
            "schema_version": PARITY_GAME_SCHEMA_VERSION,
            "protocol_id": CLASSIC7_ONUW_REFERENCE,
            "game_id": self.game_id,
            "content_profile": self.content_profile,
            "modality_profile": self.modality_profile,
            "tokens": deepcopy(self._tokens),
            "queries": deepcopy(self._queries),
            "speech_action_counts": list(self._speech_action_counts),
        }
        return validate_parity_game(game)

    def finalize_collection_audit(self) -> dict[str, Any]:
        """Return provenance sidecars that are never model input fields."""

        if len(self._label_audit) != len(self._queries):
            raise RuntimeError("every PRE query requires one label audit record")
        return {
            "game_id": self.game_id,
            "label_collector": "onuw_style_role_guess",
            "label_information": "observer_legal_private_view",
            "model_input": "public_only",
            "queries": deepcopy(self._label_audit),
            "speech_emotions": deepcopy(self._speech_emotions),
        }


__all__ = ["OnuwParityGameRecorder"]
