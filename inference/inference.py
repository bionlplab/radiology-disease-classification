from vllm import LLM, SamplingParams
import torch
import pandas as pd
from vllm.lora.request import LoRARequest
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
from prompts import simple_prompt, format_prompt2, format_prompt3, extract_think_answer, format_prompt4
import os
import re

from vllm.sampling_params import GuidedDecodingParams
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
import torch.distributed as dist
if dist.is_initialized():
    dist.destroy_process_group()

def get_available_cuda_device():
    if not torch.cuda.is_available():
        return "cuda not available"

    for i in range(4):
        if torch.cuda.get_device_properties(i).total_memory > 0:
            return f"cuda:{i}"
    
    return "no usable cuda device"

INPUT_PATH = "/prj0129/yiw4018/reasoning/final/data/train/mimic_training_2000_4000_format2_grpo.csv"
# INPUT_PATH = "/prj0129/yiw4018/reasoning/final/data/eval/nih_midrc_ground_truth.csv"
OUTPUT_FILE = f"sft.csv"
OUTPUT_PATH = "/prj0129/yiw4018/reasoning/final/result/silver_data/qwen"
MODEL_PATH = "/prj0129/yiw4018/reasoning/final/models/qwen-3b/sft_format_2000/checkpoint-999"

# Example
ENABLE_LORA = False
LORA_NAME = "length2"
LORA_ID = 120
LORA_PATH = "/prj0129/yiw4018/reasoning/final/models/llama/sft_result_only_format_1000_fmt2_grpo_2000_fmt3/checkpoint-5500"
# lora_request = LoRARequest(LORA_NAME, LORA_ID, LORA_PATH)
lora_request=None

sampling_params = SamplingParams(temperature=0.1, top_p=1, max_tokens=250)

llm = LLM(
    dtype=torch.bfloat16,  # Use bf16 for speed
    model=MODEL_PATH,
    enable_lora=ENABLE_LORA,
    max_model_len=32768,
)

os.makedirs(OUTPUT_PATH, exist_ok=True)
print("Current output path:" + OUTPUT_PATH)


def strip_and_remove_empty_lines(text):
    # Define the pattern to match "def ", "=", "code", "'''", '"""'
    pattern = r"(def\s|\=|code|'''|\"\"\")"
    
    # Search for the first occurrence of any of these patterns
    match = re.search(pattern, text)
    
    # Truncate the text before the first match
    truncated_text = text[:match.start()] if match else text
    
    # Remove empty lines and return the cleaned text
    return "\n".join(line for line in truncated_text.splitlines() if line.strip())


# Function to process a single row and return the results
def process_row(report):
    # print("**********")
    # print(report)
    prompt_str = format_prompt3(report)
    result = {
        "report": report,
    }

    for i in range(5):
        sampling_params.seed = i
        response = llm.generate(
            prompt_str,
            sampling_params,
            lora_request=lora_request
        )[0].outputs[0].text

        print("#################### RAW OUTPUT")
        print(response)
        disease, reason = extract_think_answer(response)
        
        print("~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(disease)
        print("******************************")
        # reason = strip_and_remove_empty_lines(reason)
        print(reason)
        result[f"generated_disease_{i}"] = disease
        result[f"reason_{i}"] = reason
        result[f"raw_output_{i}"] = response
    return result


df = pd.read_csv(INPUT_PATH)
df = df.head(1000)

# Initialize an empty list to store results
results = []

# Iterate over the dataframe rows
for index, row in df.iterrows():
    try:
        findings = row['report']

        print(f"Processing row {index}...")
        result = process_row(findings)
        # result['true_answer'] = row['true_answer']
        result['true_answer'] = row['output']
        results.append(result)
    except Exception as e:
        print("!!!!!!!!!!!")
        print(f"Error processing row {index}: {e}")
        print(row['report'])

# Convert results to a DataFrame
df_results = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df_results.to_csv(OUTPUT_PATH + "/" + OUTPUT_FILE, index=False)

