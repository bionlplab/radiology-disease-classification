
import os
import re
import ast
import json
import pandas as pd
from openai import AzureOpenAI

from evaluation_prompt import gpt_evaluation_prompt, output_schema

# OUTPUT_FILE = "midrc_gpt_training_source.csv"
INPUT_FILE = "/prj0129/yiw4018/reasoning/final/result/mimic/llama/grpo_2000_model_reason.csv"
OUTPUT_FILE = "/prj0129/yiw4018/reasoning/final/result/mimic/llama/grpo_2000_model_reason_evaluate.csv"
SUFFIX = "_0"
# os.makedirs(OUTPUT_PATH, exist_ok=True)


# standard
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


# Function to process a single row and return the results
def process_row(note, reasoning, disease):
    # Define the prompt template
    # prompt = generate_disease_prompt(note)
    prompt = gpt_evaluation_prompt(note, reasoning, disease)

    # while not generated_disease.startswith("[") or not generated_disease.endswith("]"):
        # Call the OpenAI GPT-4 API using the chat.completions endpoint
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful radiologist. I want you to become an evaluator of whether the reasonings are right."},
                {"role": "user", "content": prompt}
            ],
            response_format=output_schema,
            max_tokens=500,
            temperature=0,  # smaller temperature is more focused (0 is deterministic)
            top_p = 1  # default is 1, considers top_p probability mass, smaller is more deterministic
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

# Initialize an empty list to store results
results = []

# Iterate over the dataframe rows
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
        print("!!!!!!!!!!!")
        print(f"Error processing row {index}: {e}")

# Convert results to a DataFrame
df_results = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df_results.to_csv(OUTPUT_FILE, index=False)

print("Processing complete. Result saved.")