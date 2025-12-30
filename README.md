This is the code for our work "Reinforcement Learning Improves LLM Accuracy and Reasoning in Disease Classification from Radiology Reports". We propose a strategy to enchance the prediction accuracy and recover reasoning ability after supervised finetuning of LLM models. 

# Folder structure
## training
It provides the script of SFT, GRPO and the reward function.

## inference
inference.py and inference_gpt.py: disease prediction script using locally host model (Qwen, Llama, Phi3) and gpt api separately.

merge_reasoning_model.py: merge reasoning from multiple inferences

gpt_evaluator.py: script to use GPT4-o to evaluate the reasoning comprehensiveness and reasoning recall

## analysis
get_accuracy.py: get accuracy from the ensemble result

evaluation_analysis: analyze the evaluation result from gpt_evaluator.py
