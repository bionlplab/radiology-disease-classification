from vllm import LLM, SamplingParams
import torch
import pandas as pd
from vllm.lora.request import LoRARequest
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
from prompts import simple_prompt, format_prompt2, format_prompt3, extract_think_answer, format_prompt4
from openai import AzureOpenAI
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

INPUT_PATH = "/prj0129/yiw4018/reasoning/final/data/eval/mimic_ground_truth.csv"
# INPUT_PATH = "/prj0129/yiw4018/reasoning/final/data/eval/nih_midrc_ground_truth.csv"
OUTPUT_FILE = f"gpt.csv"
OUTPUT_PATH = "/prj0129/yiw4018/reasoning/final/result/mimic"
MODEL_PATH = "/prj0129/yiw4018/reasoning/final/models/phi3_mini/sft_result_only_format_2000/checkpoint-999"

os.environ["AZURE_OPENAI_KEY"] = "Fg93u942u8otZOkwwPY5Q8l6QILr7VRNYs9a8JbOmbiTT3BySGW3JQQJ99BAACYeBjFXJ3w3AAABACOGqI9S"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://yishu.openai.azure.com/openai/deployments/gpt-4o-2/chat/completions?api-version=2024-08-01-preview"

# batch
# os.environ["AZURE_OPENAI_KEY"] = "Fg93u942u8otZOkwwPY5Q8l6QILr7VRNYs9a8JbOmbiTT3BySGW3JQQJ99BAACYeBjFXJ3w3AAABACOGqI9S"
# os.environ["AZURE_OPENAI_ENDPOINT"] = "https://yishu.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"

API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT_NAME = 'gpt-4o-2'
API_VERSION = '2024-08-01-preview'

client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    base_url=API_BASE
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
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful radiologist"},
                {"role": "user", "content": prompt_str}
            ],
            max_tokens=500,
            seed=i,
            temperature=1,  # smaller temperature is more focused (0 is deterministic)
            top_p = 1  # default is 1, considers top_p probability mass, smaller is more deterministic
        )
        # Extract the generated IMPRESSION from the response
        response = response.choices[0].message.content

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

# Initialize an empty list to store results
results = []

# Iterate over the dataframe rows
for index, row in df.iterrows():
    try:
        findings = row['report']

        print(f"Processing row {index}...")
        result = process_row(findings)
        result['true_answer'] = row['true_answer']
        results.append(result)
    except Exception as e:
        print("!!!!!!!!!!!")
        print(f"Error processing row {index}: {e}")
        print(row['report'])

# Convert results to a DataFrame
df_results = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df_results.to_csv(OUTPUT_PATH + "/" + OUTPUT_FILE, index=False)

