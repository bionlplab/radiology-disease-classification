import re


def format_prompt3(note):
    return f"""You are given a clinical report. Your task is to identify any diseases mentioned in the report and format the answer as a list.
Instructions:
1. Only use diseases from this list:
    ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
2. If there is no diease, return empty list. 
3. First thinks about the reasoning process in the mind and then provides the answer. The reasoning should be based on phrases or evidence from the report.

############# Example ####################
## Report:
    Endotracheal tube terminates 6.9 cm above the carina.  The right subclavian
    line tip is at the mid SVC. The NG tube passes below the diaphragm and out of
    view.
## Answer:
    <reasoning>Support devices is found because the report mentions: 'Endotracheal tube', 'subclavian
    line' and 'NG tube'.</reasoning>
    <answer>[Support Devices]</answer>

############# Analyze the following report:
    {note}
"""


def extract_think_answer(response):
    think_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

    think = think_match.group(1).strip() if think_match else None
    answer = answer_match.group(1).strip() if answer_match else None
    return answer, think
