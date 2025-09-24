import pandas as pd
from evaluator_analysis import analyze_json_output

INPUT_PATH = "/prj0129/yiw4018/reasoning/final/result/mimic/llama/grpo_2000_model_reason_evaluate"

df = pd.read_csv(f"{INPUT_PATH}.csv")
# df = df[df['summarized_reason'].apply(lambda x: pd.notna(x) and len(x) > 0)]
df['gpt_evaluation_analysis'] = df.apply(lambda row: analyze_json_output(row['evaluation'], row['generated_disease_list'], row['true_disease_list']), axis=1)
df = df.dropna(subset=['gpt_evaluation_analysis'])
# df['reason_supported'] = df.apply(lambda row: row['gpt_evaluation_analysis']['correct_reason_count'] / row['gpt_evaluation_analysis']['total_reason_count'] 
#     if row['gpt_evaluation_analysis']['total_reason_count'] > 0 else None, axis=1
# )
# df['correct_target_ratio'] = df.apply(lambda row: 1 - row['gpt_evaluation_analysis']['missed_diseases'] / row['gpt_evaluation_analysis']['len_true'] 
#     if row['gpt_evaluation_analysis']['len_true'] > 0 else 1.0, axis=1
# )
# df['inference_correct_ratio'] = df.apply(lambda row: row['gpt_evaluation_analysis']['inference_correct'] / row['gpt_evaluation_analysis']['total_target_disease'] 
#     if row['gpt_evaluation_analysis']['total_target_disease'] > 0 else 1.0, axis=1
# )
# df['supported_generation'] = df.apply(lambda row: 1 - row['gpt_evaluation_analysis']['no_support_generation'] / row['gpt_evaluation_analysis']['len_generation'] 
#     if row['gpt_evaluation_analysis']['len_generation'] > 0 else 1.0, axis=1
# )
# df = df.dropna(subset=['reason_supported'])

df['total_target_disease'] = df.apply(lambda row: row['gpt_evaluation_analysis']['total_target_disease'], axis=1)
df['len_generation'] = df.apply(lambda row: row['gpt_evaluation_analysis']['len_generation'], axis=1)
df['len_target'] = df.apply(lambda row: row['gpt_evaluation_analysis']['len_target'], axis=1)
df['len_true'] = df.apply(lambda row: row['gpt_evaluation_analysis']['len_true'], axis=1)
df['missed_diseases'] = df.apply(lambda row: row['gpt_evaluation_analysis']['missed_diseases'], axis=1)
df['no_support_generation'] = df.apply(lambda row: row['gpt_evaluation_analysis']['no_support_generation'], axis=1)
df['total_reason_count'] = df.apply(lambda row: row['gpt_evaluation_analysis']['total_reason_count'], axis=1)
df['correct_reason_count'] = df.apply(lambda row: row['gpt_evaluation_analysis']['correct_reason_count'], axis=1)
df['target_diseases'] = df.apply(lambda row: row['gpt_evaluation_analysis']['target_diseases'], axis=1)
df['generated_disease'] = df.apply(lambda row: row['gpt_evaluation_analysis']['generated_disease'], axis=1)
df['true_disease'] = df.apply(lambda row: row['gpt_evaluation_analysis']['true_disease'], axis=1)
df.to_csv(f"{INPUT_PATH}_summary.csv")