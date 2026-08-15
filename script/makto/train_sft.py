"""Train the frozen MaKTO 7P Seer-Witch gameplay SFT dataset with QLoRA."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Sequence

import torch


TRAIN_GAME_IDS = (
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
VAL_GAME_IDS = ("game_27", "game_30", "game_34", "game_40")

EXPECTED_TASK_COUNTS = {
    "speech": 144,
    "vote": 144,
    "wolf": 60,
    "seer": 34,
    "witch": 21,
}
EXPECTED_VAL_TASK_COUNTS = {
    "speech": 28,
    "vote": 28,
    "wolf": 12,
    "seer": 7,
    "witch": 4,
}
EXPECTED_TOTAL_SAMPLES = 403
EXPECTED_TRAIN_SAMPLES = 324
EXPECTED_VAL_SAMPLES = 79
MAX_LENGTH = 2048
SEED = 42

QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16",
}
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}
TRAINING_HYPERPARAMETERS = {
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
    "seed": SEED,
    "data_seed": SEED,
    "optim": "adamw_torch",
}

_FORBIDDEN_TARGET_SEGMENTS = {
    "visual",
    "vision",
    "lm_head",
    "embed_tokens",
    "embedding",
    "embeddings",
}


class TrainingContractError(ValueError):
    """The frozen SFT input or training setup violates its contract."""


def _record_game_id(record: dict[str, Any]) -> str:
    source = record.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("game_id"), str):
        raise TrainingContractError("each record must contain source.game_id")
    return source["game_id"]


def _validate_record(record: Any, line_number: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TrainingContractError(f"line {line_number} must be a JSON object")
    game_id = _record_game_id(record)
    source = record["source"]
    event_index = source.get("event_index")
    if isinstance(event_index, bool) or not isinstance(event_index, int):
        raise TrainingContractError(
            f"line {line_number} {game_id} has invalid source.event_index"
        )
    task = record.get("task")
    if task not in EXPECTED_TASK_COUNTS:
        raise TrainingContractError(
            f"line {line_number} {game_id} has unsupported task {task!r}"
        )
    actor = record.get("actor")
    if isinstance(actor, bool) or not isinstance(actor, int):
        raise TrainingContractError(
            f"line {line_number} {game_id} has invalid actor {actor!r}"
        )
    messages = record.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or not isinstance(messages[0], dict)
        or not isinstance(messages[1], dict)
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
        or "content" not in messages[0]
        or "content" not in messages[1]
    ):
        raise TrainingContractError(
            f"line {line_number} {game_id} must contain one user and one "
            "assistant message"
        )
    return record


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise TrainingContractError(f"line {line_number} is blank")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingContractError(
                    f"line {line_number} is not valid JSON"
                ) from exc
            records.append(_validate_record(raw, line_number))
    return records


def _task_counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(record["task"] for record in records)
    return {task: counts[task] for task in EXPECTED_TASK_COUNTS}


def split_records(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(records) != EXPECTED_TOTAL_SAMPLES:
        raise TrainingContractError(
            f"expected {EXPECTED_TOTAL_SAMPLES} rows, got {len(records)}"
        )
    task_counts = Counter(record["task"] for record in records)
    if task_counts != Counter(EXPECTED_TASK_COUNTS):
        raise TrainingContractError(
            f"task counts differ: expected {EXPECTED_TASK_COUNTS}, "
            f"got {dict(task_counts)}"
        )

    train_ids = set(TRAIN_GAME_IDS)
    val_ids = set(VAL_GAME_IDS)
    if train_ids & val_ids:
        raise TrainingContractError("fixed train and validation games overlap")
    expected_ids = train_ids | val_ids
    if len(expected_ids) != 20:
        raise TrainingContractError("fixed split must contain exactly 20 games")
    actual_ids = {_record_game_id(record) for record in records}
    if actual_ids != expected_ids:
        raise TrainingContractError(
            f"game IDs differ: expected {sorted(expected_ids)}, "
            f"got {sorted(actual_ids)}"
        )

    train = [record for record in records if _record_game_id(record) in train_ids]
    validation = [
        record for record in records if _record_game_id(record) in val_ids
    ]
    if len(train) != EXPECTED_TRAIN_SAMPLES:
        raise TrainingContractError(
            f"expected {EXPECTED_TRAIN_SAMPLES} train rows, got {len(train)}"
        )
    if len(validation) != EXPECTED_VAL_SAMPLES:
        raise TrainingContractError(
            f"expected {EXPECTED_VAL_SAMPLES} validation rows, "
            f"got {len(validation)}"
        )
    val_counts = _task_counts(validation)
    if val_counts != EXPECTED_VAL_TASK_COUNTS:
        raise TrainingContractError(
            f"validation task counts differ: expected "
            f"{EXPECTED_VAL_TASK_COUNTS}, got {val_counts}"
        )
    return train, validation


def encode_record(
    record: dict[str, Any],
    tokenizer: Any,
    *,
    max_length: int = MAX_LENGTH,
) -> dict[str, list[int]]:
    user_message, assistant_message = record["messages"]
    prefix_text = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        [user_message, assistant_message],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not isinstance(prefix_text, str) or not isinstance(full_text, str):
        raise TrainingContractError("chat template rendering must return text")
    if not full_text.startswith(prefix_text):
        raise TrainingContractError(
            "full chat template rendering does not start with runtime prefix"
        )

    target_text = str(assistant_message["content"]).strip()
    if not target_text:
        raise TrainingContractError("assistant target must be non-empty")
    remainder = full_text[len(prefix_text):]
    if not remainder.startswith(target_text):
        raise TrainingContractError(
            "full chat template remainder does not start with assistant target"
        )
    suffix_text = remainder[len(target_text):]
    if not suffix_text:
        raise TrainingContractError(
            "chat template must render an assistant termination suffix"
        )

    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    if not target_ids or not suffix_ids:
        raise TrainingContractError(
            "assistant target and termination suffix must tokenize to IDs"
        )
    input_ids = list(prefix_ids) + list(target_ids) + list(suffix_ids)
    if len(input_ids) > max_length:
        source = record["source"]
        raise TrainingContractError(
            "sequence exceeds max length: "
            f"game_id={source['game_id']} "
            f"event_index={source['event_index']} "
            f"task={record['task']} actor={record['actor']} "
            f"actual_length={len(input_ids)} max_length={max_length}"
        )
    labels = [-100] * len(prefix_ids) + list(target_ids) + list(suffix_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class TokenizedDataset:
    def __init__(self, records: Sequence[dict[str, Any]], tokenizer: Any) -> None:
        self.features = [encode_record(record, tokenizer) for record in records]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.features[index]


class RightPaddingCollator:
    def __init__(self, pad_token_id: int | None) -> None:
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
            raise TrainingContractError("tokenizer.pad_token_id must be an integer")
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        features: Sequence[dict[str, list[int]]],
    ) -> dict[str, torch.Tensor]:
        if not features:
            raise TrainingContractError("cannot collate an empty batch")
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            length = len(feature["input_ids"])
            if (
                len(feature["attention_mask"]) != length
                or len(feature["labels"]) != length
            ):
                raise TrainingContractError("feature tensor lengths must match")
            padding = max_length - length
            batch["input_ids"].append(
                list(feature["input_ids"]) + [self.pad_token_id] * padding
            )
            batch["attention_mask"].append(
                list(feature["attention_mask"]) + [0] * padding
            )
            batch["labels"].append(
                list(feature["labels"]) + [-100] * padding
            )
        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in batch.items()
        }


def locate_language_model(model: torch.nn.Module) -> tuple[str, torch.nn.Module]:
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if name and name.rsplit(".", 1)[-1] == "language_model"
    ]
    if len(matches) != 1:
        raise TrainingContractError(
            f"expected exactly one language_model, found {[name for name, _ in matches]}"
        )
    return matches[0]


def validate_target_module_names(
    target_names: Sequence[str],
    language_prefix: str,
) -> None:
    if not target_names:
        raise TrainingContractError("no quantized language modules were targeted")
    prefix = language_prefix + "."
    for name in target_names:
        segments = set(name.lower().split("."))
        if not name.startswith(prefix):
            raise TrainingContractError(
                f"LoRA target is outside language_model: {name}"
            )
        forbidden = sorted(segments & _FORBIDDEN_TARGET_SEGMENTS)
        if forbidden:
            raise TrainingContractError(
                f"forbidden LoRA target {name}: {forbidden}"
            )


def discover_language_linear4bit_targets(
    model: torch.nn.Module,
    linear4bit_type: type,
) -> tuple[str, list[str]]:
    language_prefix, language_model = locate_language_model(model)
    language_module_ids = {id(module) for module in language_model.modules()}
    targets = [
        name
        for name, module in model.named_modules()
        if id(module) in language_module_ids
        and isinstance(module, linear4bit_type)
    ]
    validate_target_module_names(targets, language_prefix)
    return language_prefix, targets


def assert_only_lora_trainable(
    model: torch.nn.Module,
    language_prefix: str,
) -> dict[str, int | float]:
    trainable_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_names:
        raise TrainingContractError("PEFT model has no trainable parameters")
    for name in trainable_names:
        lower = name.lower()
        segments = set(lower.split("."))
        if "lora_" not in lower:
            raise TrainingContractError(f"non-LoRA parameter is trainable: {name}")
        if language_prefix.lower() not in lower:
            raise TrainingContractError(
                f"trainable adapter is outside language_model: {name}"
            )
        forbidden = sorted(segments & _FORBIDDEN_TARGET_SEGMENTS)
        if forbidden:
            raise TrainingContractError(
                f"forbidden trainable parameter {name}: {forbidden}"
            )

    get_parameter_counts = getattr(model, "get_nb_trainable_parameters", None)
    if not callable(get_parameter_counts):
        raise TrainingContractError(
            "PEFT model must provide get_nb_trainable_parameters()"
        )
    trainable_params, all_params = get_parameter_counts()
    if all_params <= 0:
        raise TrainingContractError("model has no parameters")
    return {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percent": 100.0 * trainable_params / all_params,
    }


def training_argument_values(
    output_dir: Path,
    *,
    max_steps: int | None,
) -> dict[str, Any]:
    if max_steps is not None and max_steps <= 0:
        raise TrainingContractError("--max-steps must be positive")
    return {
        "output_dir": str(output_dir),
        **TRAINING_HYPERPARAMETERS,
        "max_steps": -1 if max_steps is None else max_steps,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "prediction_loss_only": True,
        "remove_unused_columns": False,
        "report_to": "none",
        "do_train": True,
        "do_eval": True,
        "use_cache": False,
    }


def validate_trainer_loss_semantics(trainer: Any) -> None:
    if getattr(trainer, "model_accepts_loss_kwargs", None) is not False:
        raise TrainingContractError(
            "current training contract requires per-sample mean loss with "
            "model_accepts_loss_kwargs=False"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    packages = ("torch", "transformers", "accelerate", "peft", "bitsandbytes")
    return {name: importlib.metadata.version(name) for name in packages}


def build_manifest(
    *,
    model_name: str,
    data_path: Path,
    data_sha256: str,
    targeted_module_count: int,
    parameter_stats: dict[str, int | float],
    max_steps: int | None,
    package_versions: dict[str, str],
) -> dict[str, Any]:
    return {
        "base_model": model_name,
        "data_path": str(data_path.resolve()),
        "data_sha256": data_sha256,
        "train_game_ids": list(TRAIN_GAME_IDS),
        "val_game_ids": list(VAL_GAME_IDS),
        "train_sample_count": EXPECTED_TRAIN_SAMPLES,
        "val_sample_count": EXPECTED_VAL_SAMPLES,
        "task_counts": dict(EXPECTED_TASK_COUNTS),
        "max_length": MAX_LENGTH,
        "enable_thinking": False,
        "quantization": dict(QUANTIZATION_CONFIG),
        "lora": dict(LORA_CONFIG),
        "targeted_module_count": targeted_module_count,
        **parameter_stats,
        "hyperparameters": {
            **TRAINING_HYPERPARAMETERS,
            "max_steps": max_steps,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
        },
        "package_versions": dict(package_versions),
        "seed": SEED,
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _prepare_qlora_model(model_name: str):
    from bitsandbytes.nn import Linear4bit
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available():
        raise TrainingContractError("4-bit QLoRA training requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise TrainingContractError("4-bit QLoRA training requires CUDA BF16")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
    )
    if model.__class__.__name__ != "Qwen3_5ForConditionalGeneration":
        raise TrainingContractError(
            f"expected Qwen3_5ForConditionalGeneration, got {model.__class__.__name__}"
        )
    if not getattr(model, "is_loaded_in_4bit", False):
        raise TrainingContractError("model did not load as bitsandbytes 4-bit")
    device_map = getattr(model, "hf_device_map", {})
    if any(str(device).lower() in {"cpu", "disk"} for device in device_map.values()):
        raise TrainingContractError(f"CPU/disk model offload is forbidden: {device_map}")
    text_config = getattr(model.config, "text_config", None)
    if text_config is None or not hasattr(text_config, "use_cache"):
        raise TrainingContractError("Qwen3.5 text_config.use_cache is unavailable")
    text_config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    if not getattr(model, "is_gradient_checkpointing", False):
        raise TrainingContractError("gradient checkpointing was not enabled")

    language_prefix, target_names = discover_language_linear4bit_targets(
        model,
        Linear4bit,
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_names,
    )
    model = get_peft_model(model, lora_config)
    if not isinstance(model, PeftModel):
        raise TrainingContractError("get_peft_model did not return a PEFT model")
    parameter_stats = assert_only_lora_trainable(model, language_prefix)
    return model, target_names, parameter_stats


def train(
    *,
    model_name: str,
    data_path: Path,
    output_dir: Path,
    max_steps: int | None,
) -> None:
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    records = load_records(data_path)
    train_records, val_records = split_records(records)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        raise TrainingContractError("Qwen3.5 tokenizer has no pad_token_id")
    train_dataset = TokenizedDataset(train_records, tokenizer)
    val_dataset = TokenizedDataset(val_records, tokenizer)

    model, target_names, parameter_stats = _prepare_qlora_model(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "training_manifest.json"
    manifest = build_manifest(
        model_name=model_name,
        data_path=data_path,
        data_sha256=_sha256(data_path),
        targeted_module_count=len(target_names),
        parameter_stats=parameter_stats,
        max_steps=max_steps,
        package_versions=_package_versions(),
    )
    _write_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "targeted_module_count": len(target_names),
                **parameter_stats,
            },
            sort_keys=True,
        )
    )

    training_args = TrainingArguments(
        **training_argument_values(output_dir, max_steps=max_steps)
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=RightPaddingCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    validate_trainer_loss_semantics(trainer)
    trainer.train()
    best_checkpoint = trainer.state.best_model_checkpoint
    best_eval_loss = trainer.state.best_metric
    if not isinstance(best_checkpoint, str) or best_eval_loss is None:
        raise TrainingContractError(
            "Trainer did not produce a best eval_loss checkpoint"
        )
    best_adapter = output_dir / "best_adapter"
    trainer.model.save_pretrained(best_adapter, safe_serialization=True)
    manifest.update(
        {
            "best_checkpoint": best_checkpoint,
            "best_eval_loss": float(best_eval_loss),
            "best_adapter": str(best_adapter),
            "epochs_completed": trainer.state.epoch,
        }
    )
    _write_manifest(manifest_path, manifest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train Qwen3.5-9B on frozen MaKTO gameplay SFT data."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)
    train(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
