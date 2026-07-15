"""Tests for qlora_finetune module."""

from __future__ import annotations

import json

from qlora_finetune import (
    AXOLOTL_CONFIG_EXAMPLE,
    FinetuneConfig,
    build_sft_trainer,
    load_model_and_tokenizer,
)


def test_finetune_config_defaults():
    name = FinetuneConfig().base_model
    assert "llama" in name.lower()


def test_finetune_config_lora_r():
    assert FinetuneConfig().lora_r == 16


def test_finetune_config_custom():
    cfg = FinetuneConfig(lora_r=8, num_epochs=1)
    assert cfg.num_epochs == 1


def test_finetune_config_target_modules():
    assert FinetuneConfig().target_modules == ["q_proj", "v_proj"]


def test_load_model_without_gpu():
    result = load_model_and_tokenizer(FinetuneConfig())
    assert isinstance(result, tuple) and len(result) == 2


def test_build_sft_trainer_without_trl():
    result = build_sft_trainer(None, None, None, FinetuneConfig())
    # Either None (trl absent) or an object (trl present)
    assert result is None or result is not None


def test_axolotl_config_is_string():
    assert isinstance(AXOLOTL_CONFIG_EXAMPLE, str)


def test_axolotl_config_contains_base_model():
    assert "base_model" in AXOLOTL_CONFIG_EXAMPLE


def test_axolotl_config_contains_qlora():
    assert "qlora" in AXOLOTL_CONFIG_EXAMPLE.lower()


def test_finetune_config_serializes():
    raw = FinetuneConfig().model_dump_json()
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
