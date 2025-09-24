import re
import random


def normalize_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip(" ")
        start = value.find("[")
        end = value.find("]")
        if end == -1:
            end = len(value)
        value = value[start+1:end]
        items = [item.strip().strip("'").strip('"') for item in value.split(",") if item.strip()]
        
        return [item.lower() for item in items if item.lower() != "normal"]
    
    return []


def strip_and_remove_empty_lines(text):
    if text is None:
        return ""
    pattern = r"(def\s|\=|code|'''|\"\"\")"

    match = re.search(pattern, text)
    
    truncated_text = text[:match.start()] if match else text
    
    return "\n".join(line for line in truncated_text.splitlines() if line.strip())


def extract_think_answer(response):
    think_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

    think = think_match.group(1).strip() if think_match else None
    answer = answer_match.group(1).strip() if answer_match else None
    return answer, think


def soft_format_reward_func(response) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    matches = re.match(pattern, response)
    return 1 if matches else 0.0


def format_reward(completions, output, reason, prompts, **kwargs):
    rewards = []
    for completion, truth, true_reason in zip(completions, output, reason):
        try:
            format_reward = soft_format_reward_func(completion)
            rewards += [format_reward]
        except Exception as e:
            print("ERROR is:")
            print(e)
            rewards += [0]
    return rewards


def pr_must_reason_reward(completions, output, reason, prompts, **kwargs):
    rewards = []
    for completion, truth, true_reason in zip(completions, output, reason):
        try:
            print("RAW OUTPUT ##########################")
            print(completion)
            answer, answer_reason = extract_think_answer(completion)
            should_reward = False if answer is None else True
            answer = set(normalize_list(answer))
            truth = set(normalize_list(truth))
            intersection = answer & truth
            precision = 1 if len(answer) == 0 else len(intersection) / len(answer)
            recall = 1 if len(truth) == 0 else len(intersection) / len(truth)
            if recall == 0:
                precision = 0
            if precision == 0:
                recall = 0
            if not should_reward:
                precision = 0
                recall = 0
            answer_reason = strip_and_remove_empty_lines(answer_reason)
            accuracy_reward = 0.5 * precision + 0.5 * recall
            def check_string(s):
                if s is None or len(s) == 0:
                    return 0
                # Check for brackets
                if '[' in s or ']' in s:
                    return 0
                
                # Split by commas and check each segment
                segments = s.split(r'[,.]')
                for segment in segments:
                    words = segment.strip().split()
                    if len(words) <= 2:
                        return 0
                
                return 1

            format_reward = check_string(answer_reason)
            rewards += [0.8 * accuracy_reward + 0.2 * format_reward]
        except Exception as e:
            print("ERROR is:")
            print(e)
            rewards += [0]
    return rewards

    rewards = []
    for completion, truth, true_reason in zip(completions, output, reason):
        try:
            print("RAW OUTPUT ##########################")
            print(completion)
            answer, answer_reason = extract_think_answer(completion)
            should_reward = False if answer is None else True
            answer = set(normalize_list(answer))
            truth = set(normalize_list(truth))
            intersection = answer & truth
            precision = 1 if len(answer) == 0 else len(intersection) / len(answer)
            recall = 1 if len(truth) == 0 else len(intersection) / len(truth)
            if recall == 0:
                precision = 0
            if precision == 0:
                recall = 0
            if not should_reward:
                precision = 0
                recall = 0
            answer_reason = strip_and_remove_empty_lines(answer_reason)
            if random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) > 0:
                print("ANSWER #####################################")
                print(answer)
                print("TRUE ANSWER -----------------------------")
                print(truth)
                print("ANSWER REASON -----------------------------")
                print(answer_reason)
                print("TRUE REASON -----------------------------")
                print(true_reason)
                print("PRECISON, RECALL --------------------------")
                print(precision)
                print(recall)
            accuracy_reward = 0.5 * precision + 0.5 * recall
            format_reward = 0 if len(answer_reason) < 100 else 1
            print("FORMAT REWARD ----------------------")
            print(format_reward)
            rewards += [0.5 * accuracy_reward + 0.5 * format_reward]
        except Exception as e:
            print("ERROR is:")
            print(e)
            rewards += [0]
    return rewards