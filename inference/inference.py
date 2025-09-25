from vllm import LLM, SamplingParams
import torch
import pandas as pd
from prompts import format_prompt3, extract_think_answer
import os


INPUT_PATH = ""
OUTPUT_FILE = ""
OUTPUT_PATH = ""
MODEL_PATH = ""

ENABLE_LORA = False
LORA_NAME = ""
LORA_ID = 120
LORA_PATH = ""
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


def process_row(report):
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
        
        result[f"generated_disease_{i}"] = disease
        result[f"reason_{i}"] = reason
        result[f"raw_output_{i}"] = response
    return result


df = pd.read_csv(INPUT_PATH)
df = df.head(1000)

results = []

for index, row in df.iterrows():
    try:
        findings = row['report']

        print(f"Processing row {index}...")
        result = process_row(findings)
        result['true_answer'] = row['true_answer']
        result['true_answer'] = row['output']
        results.append(result)
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        print(row['report'])

df_results = pd.DataFrame(results)

df_results.to_csv(OUTPUT_PATH + "/" + OUTPUT_FILE, index=False)

