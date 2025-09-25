from vllm import LLM, SamplingParams
import torch
import pandas as pd


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


INPUT_FILE = ""
OUTPUT_FILE = ""
MODEL_PATH = ""


ENABLE_LORA = False
LORA_NAME = ""
LORA_ID = 116
LORA_PATH = ""
lora_request=None

sampling_params = SamplingParams(temperature=0.1, top_p=1, max_tokens=250)

llm = LLM(
    dtype=torch.bfloat16,
    model=MODEL_PATH,
    enable_lora=ENABLE_LORA,
)


def process_row(reason_0, reason_1, reason_2, reason_3, reason_4, disease_ensemble):
    prompt = merge_reason_with_disease_enemble(reason_0, reason_1, reason_2, reason_3, reason_4, disease_ensemble)

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

results = []

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
        print(f"Error processing row {index}: {e}")

df_results = pd.DataFrame(results)

df_results.to_csv(OUTPUT_FILE, index=False)

print("Processing complete. Result saved.")

