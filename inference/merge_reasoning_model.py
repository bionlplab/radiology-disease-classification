from vllm import LLM, SamplingParams
import torch
import pandas as pd
from vllm.lora.request import LoRARequest
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
from prompts import simple_prompt, format_prompt2, format_prompt3, extract_think_answer, format_prompt4
import os
import re


def merge_reason_with_disease_enemble(reason_0, reason_1, reason_2, reason_3, reason_4, disease_ensemble):
    return f"""
    You are a radiology specialist. You are provided with five reasonings of radiology report prediction of the same 
    report. You are also given a final disease list, which is based on majority voting of the five predictions.
    Your taks is to summarize the five reasonings and get an overral reasoning report.

    ### Instructions
        1. The reasoning should be focued on explaining the final disease list. So you don't need to include all points
            from the provided reasonings.
        2. The reasoning should mostly be finding evidence from original report.
        3. The provided reasonings may not be of high quality, so you need to paraphrase and summarize them.
        4. Summarize the report naturally, so do not show that you are summarizing. Output is a natural reasoning report.
        5. DO NOT make up any evidence. ONLY use evidence from the provided reasonings. If all reasonings are empty, return empty output as well.

    ### Input:
        ##  reason0: {reason_0}
        ##  reason1: {reason_1}
        ##  reason2: {reason_2}
        ##  reason3: {reason_3}
        ##  reason4: {reason_4}
        ## disease_list: {disease_ensemble}
    ### Your summarized reasoning:
    """

# OUTPUT_FILE = "midrc_gpt_training_source.csv"
INPUT_FILE = "/prj0129/yiw4018/reasoning/final/result/mimic/llama/grpo_2000.csv"
OUTPUT_FILE = "/prj0129/yiw4018/reasoning/final/result/mimic/llama/grpo_2000_model_reason.csv"
MODEL_PATH = "/prj0129/yiw4018/george/meta-llama/Llama-3.1-8B"
# os.makedirs(OUTPUT_PATH, exist_ok=True)


ENABLE_LORA = False
LORA_NAME = "sft_result_only_format_1000_fmt2_grpo_2000_fmt3"
LORA_ID = 116
LORA_PATH = "/prj0129/yiw4018/reasoning/final/models/llama/sft_result_only_format_1000_fmt2_grpo_2000_fmt3/checkpoint-5500"
lora_request=None

sampling_params = SamplingParams(temperature=0.1, top_p=1, max_tokens=250)

llm = LLM(
    dtype=torch.bfloat16,  # Use bf16 for speed
    model=MODEL_PATH,
    enable_lora=ENABLE_LORA,
)


# Function to process a single row and return the results
def process_row(reason_0, reason_1, reason_2, reason_3, reason_4, disease_ensemble):
    # Define the prompt template
    # prompt = generate_disease_prompt(note)
    prompt = merge_reason_with_disease_enemble(reason_0, reason_1, reason_2, reason_3, reason_4, disease_ensemble)

    # while not generated_disease.startswith("[") or not generated_disease.endswith("]"):
        # Call the OpenAI GPT-4 API using the chat.completions endpoint
    try:
        response = llm.generate(
            prompt,
            sampling_params,
            lora_request=lora_request
        )[0].outputs[0].text

        print("#################### RAW OUTPUT")
        print(response)
        print(response)
        result = {
            "summarized_reason": response,
        }
        return result
    except Exception as e:
        print(e)
        result = {
            "summarized_reason": None,
        }
        return None


df = pd.read_csv(INPUT_FILE)
print("CSV file read successfully.")

# Initialize an empty list to store results
results = []

# Iterate over the dataframe rows
for index, row in df.iterrows():
    try:
        reason0 = row[f"reason_0"]
        reason1 = row[f"reason_1"]
        reason2 = row[f"reason_2"]
        reason3 = row[f"reason_3"]
        reason4 = row[f"reason_4"]
        disease_list = row["generated_disease_list"]

        print(f"Processing row {index}...")
        result = process_row(reason0, reason1, reason2, reason3, reason4, disease_list)
        result['report'] = row['report']
        result['reason_0'] = reason0
        result['reason_1'] = reason1
        result['reason_2'] = reason2
        result['reason_3'] = reason3
        result['reason_4'] = reason4
        result["generated_disease_list"] = row["generated_disease_list"]
        result['true_disease_list'] = row['true_answer']
        results.append(result)
    except Exception as e:
        print("!!!!!!!!!!!")
        print(f"Error processing row {index}: {e}")

# Convert results to a DataFrame
df_results = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df_results.to_csv(OUTPUT_FILE, index=False)

print("Processing complete. Result saved.")

