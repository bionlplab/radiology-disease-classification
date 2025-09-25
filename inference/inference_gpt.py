import pandas as pd
from prompts import format_prompt3, extract_think_answer
from openai import AzureOpenAI
import os
import re

INPUT_PATH = ""
OUTPUT_FILE = ""
OUTPUT_PATH = ""
MODEL_PATH = ""

os.environ["AZURE_OPENAI_KEY"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""


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


def process_row(report):
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
            temperature=1,
            top_p = 1
        )
        # Extract the generated IMPRESSION from the response
        response = response.choices[0].message.content

        print("#################### RAW OUTPUT")
        print(response)
        disease, reason = extract_think_answer(response)
        
        result[f"generated_disease_{i}"] = disease
        result[f"reason_{i}"] = reason
        result[f"raw_output_{i}"] = response
    return result


df = pd.read_csv(INPUT_PATH)

results = []

for index, row in df.iterrows():
    try:
        findings = row['report']

        print(f"Processing row {index}...")
        result = process_row(findings)
        result['true_answer'] = row['true_answer']
        results.append(result)
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        print(row['report'])

df_results = pd.DataFrame(results)

df_results.to_csv(OUTPUT_PATH + "/" + OUTPUT_FILE, index=False)

