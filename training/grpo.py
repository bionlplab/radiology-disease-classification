import torch._dynamo
torch._dynamo.config.optimize_ddp = False


import numpy
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from reward import pr_must_reason_reward
from transformers import AutoTokenizer
import torch

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

from peft import LoraConfig, TaskType

# LR=1e-4
lora_config = LoraConfig(
    r=16,
    target_modules=["k_proj", "q_proj", "v_proj", "output_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=32,
    lora_dropout=0.05,
)


# LR=1e-6

MODEL = "/prj0129/yiw4018/reasoning/final/models/llama/sft_result_only_format_2000_fmt2_not_full/checkpoint-500"
tokenizer = AutoTokenizer.from_pretrained("/prj0129/yiw4018/reasoning/final/models/llama/sft_result_only_format_2000_fmt2_not_full/checkpoint-500")
OUTPUT_DIR = "/prj0129/yiw4018/reasoning/final/models/llama/experimental/length"
DATA_PATH = "/prj0129/yiw4018/reasoning/final/data/train/revised_data/mimic_training_2000_grpo_format3.csv"

tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("csv", data_files=DATA_PATH, split="train")
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    logging_dir=OUTPUT_DIR + "/logging",
    logging_steps=10,
    max_prompt_length=512,
    max_completion_length=256,
    num_generations=8,
    per_device_train_batch_size=4,
    torch_empty_cache_steps=1,  # lower speed
    gradient_accumulation_steps=1,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
     
    torch_compile=True,
    model_init_kwargs = dict(
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ),
)

trainer = GRPOTrainer(
    model=MODEL,
    reward_funcs=pr_must_reason_reward,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)
trainer.train()