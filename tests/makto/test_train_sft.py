import json

import pytest
import torch

from script.makto.train_sft import (
    EXPECTED_TASK_COUNTS,
    EXPECTED_TRAIN_SAMPLES,
    EXPECTED_VAL_SAMPLES,
    EXPECTED_VAL_TASK_COUNTS,
    MAX_LENGTH,
    SEED,
    TRAIN_GAME_IDS,
    VAL_GAME_IDS,
    RightPaddingCollator,
    TrainingContractError,
    assert_only_lora_trainable,
    build_manifest,
    discover_language_linear4bit_targets,
    encode_record,
    load_records,
    split_records,
    training_argument_values,
    validate_target_module_names,
    validate_trainer_loss_semantics,
)


def _record(game_id, task, event_index, *, target="answer"):
    return {
        "source": {
            "dataset": "makto",
            "revision": "fixed",
            "split": "train",
            "setting": "7_player_game/seer_witch",
            "game_id": game_id,
            "event_index": event_index,
        },
        "task": task,
        "actor": 1,
        "role": "Werewolf",
        "candidate_snapshot": None,
        "messages": [
            {"role": "user", "content": f"prompt-{game_id}-{event_index}"},
            {"role": "assistant", "content": target},
        ],
    }


def _fixed_records():
    records = []
    event_index = 0
    train_counts = {
        task: EXPECTED_TASK_COUNTS[task] - EXPECTED_VAL_TASK_COUNTS[task]
        for task in EXPECTED_TASK_COUNTS
    }
    for game_ids, task_counts in (
        (TRAIN_GAME_IDS, train_counts),
        (VAL_GAME_IDS, EXPECTED_VAL_TASK_COUNTS),
    ):
        game_cursor = 0
        for task, count in task_counts.items():
            for _ in range(count):
                records.append(
                    _record(
                        game_ids[game_cursor % len(game_ids)],
                        task,
                        event_index,
                    )
                )
                event_index += 1
                game_cursor += 1
    return records


class FakeTokenizer:
    pad_token_id = 99

    def __init__(self):
        self.template_calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        self.template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        prefix = f"<u>{messages[0]['content']}</u><a>"
        if add_generation_prompt:
            return prefix
        return prefix + str(messages[1]["content"]) + "</a>"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(character) for character in text]


def test_fixed_whole_game_split_is_exact_disjoint_and_order_preserving(tmp_path):
    records = _fixed_records()
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    loaded = load_records(data_path)
    train, validation = split_records(loaded)

    assert tuple(TRAIN_GAME_IDS) == (
        "game_21",
        "game_22",
        "game_23",
        "game_24",
        "game_25",
        "game_26",
        "game_28",
        "game_29",
        "game_31",
        "game_32",
        "game_33",
        "game_35",
        "game_36",
        "game_37",
        "game_38",
        "game_39",
    )
    assert tuple(VAL_GAME_IDS) == ("game_27", "game_30", "game_34", "game_40")
    assert set(TRAIN_GAME_IDS).isdisjoint(VAL_GAME_IDS)
    assert set(TRAIN_GAME_IDS) | set(VAL_GAME_IDS) == {
        f"game_{number}" for number in range(21, 41)
    }
    assert len(train) == EXPECTED_TRAIN_SAMPLES
    assert len(validation) == EXPECTED_VAL_SAMPLES
    assert {
        task: sum(record["task"] == task for record in validation)
        for task in EXPECTED_TASK_COUNTS
    } == EXPECTED_VAL_TASK_COUNTS
    assert train == [
        record
        for record in loaded
        if record["source"]["game_id"] in TRAIN_GAME_IDS
    ]
    assert validation == [
        record
        for record in loaded
        if record["source"]["game_id"] in VAL_GAME_IDS
    ]

    with pytest.raises(TrainingContractError, match="expected 403 rows"):
        split_records(loaded[:-1])
    wrong_counts = list(loaded)
    wrong_counts[0] = {**wrong_counts[0], "task": "vote"}
    with pytest.raises(TrainingContractError, match="task counts differ"):
        split_records(wrong_counts)


def test_labels_mask_prefix_and_supervise_exact_target_and_suffix():
    tokenizer = FakeTokenizer()
    target = '{"action_index":3}'
    record = _record("game_21", "vote", 17, target=target)

    feature = encode_record(record, tokenizer)
    prefix_text = f"<u>{record['messages'][0]['content']}</u><a>"
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    suffix_ids = tokenizer.encode("</a>", add_special_tokens=False)

    assert feature["input_ids"] == prefix_ids + target_ids + suffix_ids
    assert feature["labels"][:len(prefix_ids)] == [-100] * len(prefix_ids)
    assert feature["labels"][len(prefix_ids):] == target_ids + suffix_ids
    assert feature["attention_mask"] == [1] * len(feature["input_ids"])
    assert "".join(chr(token) for token in target_ids) == target
    assert len(tokenizer.template_calls) == 2
    assert all(
        call["enable_thinking"] is False
        and call["tokenize"] is False
        for call in tokenizer.template_calls
    )
    assert tokenizer.template_calls[0]["add_generation_prompt"] is True
    assert tokenizer.template_calls[1]["add_generation_prompt"] is False


def test_right_padding_masks_attention_and_labels():
    collator = RightPaddingCollator(pad_token_id=99)
    batch = collator(
        [
            {
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
                "labels": [-100, 2],
            },
            {
                "input_ids": [3, 4, 5, 6],
                "attention_mask": [1, 1, 1, 1],
                "labels": [-100, 4, 5, 6],
            },
        ]
    )

    assert batch["input_ids"].tolist() == [[1, 2, 99, 99], [3, 4, 5, 6]]
    assert batch["attention_mask"].tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]
    assert batch["labels"].tolist() == [
        [-100, 2, -100, -100],
        [-100, 4, 5, 6],
    ]


def test_length_ceiling_never_truncates_and_reports_sample_identity():
    tokenizer = FakeTokenizer()
    within_limit = _record("game_21", "speech", 8)
    within_limit["messages"][0]["content"] = "x" * 2000
    feature = encode_record(within_limit, tokenizer)
    assert len(feature["input_ids"]) <= MAX_LENGTH

    too_long = _record("game_40", "seer", 99)
    too_long["actor"] = 6
    too_long["messages"][0]["content"] = "x" * 2048
    with pytest.raises(TrainingContractError) as error:
        encode_record(too_long, tokenizer)
    message = str(error.value)
    assert "game_id=game_40" in message
    assert "event_index=99" in message
    assert "task=seer" in message
    assert "actor=6" in message
    assert "actual_length=" in message
    assert "max_length=2048" in message


class FakeLinear4bit(torch.nn.Linear):
    pass


class FakeQuantizedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.block = FakeLinear4bit(2, 2)
        self.model.language_model.embed_tokens = torch.nn.Embedding(2, 2)
        self.model.visual = torch.nn.Module()
        self.model.visual.projection = FakeLinear4bit(2, 2)
        self.lm_head = FakeLinear4bit(2, 2)


def test_lora_target_discovery_accepts_only_language_linear4bit_modules():
    prefix, targets = discover_language_linear4bit_targets(
        FakeQuantizedModel(),
        FakeLinear4bit,
    )
    assert prefix == "model.language_model"
    assert targets == ["model.language_model.block"]

    validate_target_module_names(
        ["model.language_model.layers.0.self_attn.q_proj"],
        prefix,
    )
    with pytest.raises(TrainingContractError, match="outside language_model"):
        validate_target_module_names(["model.visual.projection"], prefix)
    with pytest.raises(TrainingContractError, match="forbidden LoRA target"):
        validate_target_module_names(["model.language_model.lm_head"], prefix)
    with pytest.raises(TrainingContractError, match="forbidden LoRA target"):
        validate_target_module_names(
            ["model.language_model.embed_tokens"],
            prefix,
        )


class FakeAdapterModel(torch.nn.Module):
    def __init__(self, *, unsafe=False, parameter_counts=(123, 1000)):
        super().__init__()
        self.parameter_counts = parameter_counts
        self.parameter_count_calls = 0
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.block = torch.nn.Module()
        self.model.language_model.block.lora_A = torch.nn.Linear(
            2, 2, bias=False
        )
        self.model.language_model.block.base_weight = torch.nn.Parameter(
            torch.zeros(2, 2),
            requires_grad=unsafe,
        )

    def get_nb_trainable_parameters(self):
        self.parameter_count_calls += 1
        return self.parameter_counts


def test_trainable_parameters_must_be_language_lora_only():
    safe = FakeAdapterModel()
    stats = assert_only_lora_trainable(safe, "model.language_model")
    assert stats["trainable_params"] == 123
    assert stats["all_params"] == 1000
    assert stats["trainable_percent"] == 12.3
    assert safe.parameter_count_calls == 1

    unsafe = FakeAdapterModel(unsafe=True)
    with pytest.raises(TrainingContractError, match="non-LoRA parameter"):
        assert_only_lora_trainable(unsafe, "model.language_model")


def test_trainer_requires_per_sample_mean_loss_semantics():
    class FakeTrainer:
        def __init__(self, model_accepts_loss_kwargs):
            self.model_accepts_loss_kwargs = model_accepts_loss_kwargs

    validate_trainer_loss_semantics(FakeTrainer(False))
    with pytest.raises(
        TrainingContractError,
        match="per-sample mean loss.*model_accepts_loss_kwargs=False",
    ):
        validate_trainer_loss_semantics(FakeTrainer(True))


def test_manifest_and_training_arguments_freeze_audited_configuration(tmp_path):
    stats = {
        "trainable_params": 123,
        "all_params": 456,
        "trainable_percent": 26.973684210526315,
    }
    versions = {
        "torch": "2.13.0+cu130",
        "transformers": "5.14.1",
        "accelerate": "1.14.0",
        "peft": "0.20.0",
        "bitsandbytes": "0.50.1",
    }
    manifest = build_manifest(
        model_name="/models/Qwen3.5-9B",
        data_path=tmp_path / "data.jsonl",
        data_sha256="abc123",
        targeted_module_count=77,
        parameter_stats=stats,
        max_steps=None,
        package_versions=versions,
    )
    args = training_argument_values(tmp_path / "output", max_steps=None)

    assert manifest["train_game_ids"] == list(TRAIN_GAME_IDS)
    assert manifest["val_game_ids"] == list(VAL_GAME_IDS)
    assert manifest["train_sample_count"] == 324
    assert manifest["val_sample_count"] == 79
    assert manifest["task_counts"] == EXPECTED_TASK_COUNTS
    assert manifest["max_length"] == 2048
    assert manifest["enable_thinking"] is False
    assert manifest["quantization"] == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    }
    assert manifest["lora"] == {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    assert manifest["package_versions"] == versions
    assert "timestamp" not in manifest
    assert manifest["hyperparameters"] == {
        "num_train_epochs": 5,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "warmup_steps": 0.10,
        "lr_scheduler_type": "linear",
        "max_grad_norm": 1.0,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "seed": 42,
        "data_seed": 42,
        "optim": "adamw_torch",
        "max_steps": None,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
    }
    assert args["seed"] == SEED
    assert args["data_seed"] == SEED
    assert args["num_train_epochs"] == 5
    assert args["per_device_train_batch_size"] == 1
    assert args["per_device_eval_batch_size"] == 1
    assert args["gradient_accumulation_steps"] == 8
    assert args["learning_rate"] == 1e-4
    assert args["warmup_steps"] == 0.10
    assert "warmup_ratio" not in args
    assert manifest["hyperparameters"]["warmup_steps"] == 0.10
    assert "warmup_ratio" not in manifest["hyperparameters"]
    assert args["optim"] == "adamw_torch"
    assert args["eval_strategy"] == "epoch"
    assert args["save_strategy"] == "epoch"
    assert args["load_best_model_at_end"] is True
    assert args["metric_for_best_model"] == "eval_loss"
    assert args["greater_is_better"] is False
    assert args["max_steps"] == -1
    assert training_argument_values(
        tmp_path / "smoke", max_steps=1
    )["max_steps"] == 1
