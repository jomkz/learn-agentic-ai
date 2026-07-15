"""QLoRA fine-tuning pipeline using TRL SFTTrainer. Requires GPU and ml extras."""

from __future__ import annotations

from pydantic import BaseModel


class FinetuneConfig(BaseModel):
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "v_proj"]
    load_in_4bit: bool = True
    output_dir: str = "./output"
    num_epochs: int = 3
    per_device_batch_size: int = 2
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4


def load_model_and_tokenizer(config: FinetuneConfig):
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError:
        return (None, None)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    return (model, tokenizer)


def build_sft_trainer(model, tokenizer, dataset, config: FinetuneConfig):
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError:
        return None

    return SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.per_device_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation,
            learning_rate=config.learning_rate,
            logging_steps=10,
            save_strategy="epoch",
        ),
        train_dataset=dataset,
        tokenizer=tokenizer,
    )


AXOLOTL_CONFIG_EXAMPLE: str = """\
base_model: meta-llama/Llama-3.2-3B-Instruct
load_in_4bit: true
adapter: qlora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
datasets:
  - path: data/train.jsonl
    type: alpaca
output_dir: ./axolotl-output
num_epochs: 3
micro_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2e-4
"""


if __name__ == "__main__":
    print(AXOLOTL_CONFIG_EXAMPLE)
    print("Run with: axolotl train config.yaml")
