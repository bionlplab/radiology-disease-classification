
import os
import pandas as pd
from openai import AzureOpenAI

from evaluation_prompt import gpt_evaluation_prompt, output_schema


INPUT_FILE = ""
OUTPUT_FILE = ""
SUFFIX = "_0"


# standard
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


def process_row(note, reasoning, disease):
    prompt = gpt_evaluation_prompt(note, reasoning, disease)

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful radiologist. I want you to become an evaluator of whether the reasonings are right."},
                {"role": "user", "content": prompt}
            ],
            response_format=output_schema,
            max_tokens=500,
            temperature=0,
            top_p = 1
        )
        # Extract the generated IMPRESSION from the response
        response = response.choices[0].message.content
        print(response)
        result = {
            "evaluation": response,
        }
        return result
    except Exception as e:
        print(e)
        result = {
            "evaluation": None,
        }
        return None


df = pd.read_csv(INPUT_FILE)
def extract_before_empty_line(text):
    result = text
    try:
        result = text.split("\n\n")[0]
    except:
        print(text)
    return result
df['summarized_reason'] = df['summarized_reason'].apply(extract_before_empty_line)

print("CSV file read successfully.")

results = []

for index, row in df.iterrows():
    try:
        report = row['report']
        reasoning = row[f"summarized_reason"]
        result = row[f"generated_disease_list"]

        print(f"Processing row {index}...")
        result = process_row(report, reasoning, result)
        result['report'] = row['report']
        result[f"summarized_reason"] = row[f"summarized_reason"]
        result[f"generated_disease_list"] = row[f"generated_disease_list"]
        result['true_disease_list'] = row['true_disease_list']
        results.append(result)
    except Exception as e:
        print(f"Error processing row {index}: {e}")

df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_FILE, index=False)

print("Processing complete. Result saved.")