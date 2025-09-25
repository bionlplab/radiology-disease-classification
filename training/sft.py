from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer
import torch

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

SAVE_STEP = 500
tokenizer = AutoTokenizer.from_pretrained("")
output_dir=""
model=""
DATA_FILE = ""

tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset("csv", data_files=DATA_FILE, split="train")
training_args = SFTConfig(
    output_dir=output_dir,
    max_seq_length=1024,
    model_init_kwargs = dict(
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ),
    bf16=True,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    save_steps=SAVE_STEP,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()