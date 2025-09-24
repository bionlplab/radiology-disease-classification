import re


def simple_prompt(note):
    return f"""
You are given a clinical report. Your task is to identify any diseases mentioned in the report and format the answer as a list ([disease1, disease2, ...]). 
If there is no diease, return empty list. Return this list on first line, followed by your reasoning starting the second line.

Only use diseases from this list, there are 13 diseases. If there are no diseases, output empty list
['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

    Analyze the following report, output the list in the first line, provide your reasoning starting second line:
    {note}
    Output:
"""


def gpt_prompt(note):
    return f"""
    You are given a clinical report. Your task is to identify any diseases mentioned in the report and list them. The result list should be formatted exactly as [disease1, disease2, ...] and be on a single line.

    Only use diseases from this list:
    ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

    Guidelines:
    Be careful about negation. E.g., do not include Pneumonia if the text says "no Pneumonia"
    Only include diseases that are certain. Entities that are 'likely', 'possibly' should not be returned.   
    
    **Example Reports and Outputs:**

    1. Report: 'Lung volumes are markedly low.  This results in exaggeration of the cardiac\n silhouette size which is borderline enlarged.  Mediastinal and hilar contours\n are unremarkable.  Crowding of the bronchovascular structures is present\n without overt pulmonary edema.  Patchy and linear opacities in the lung bases\n most likely are reflective of atelectasis.  No focal consolidation, pleural\n effusion or pneumothorax is identified.  No acute osseous abnormality is\n visualized.'
    Output:
    ['Lung Opacity']
    Reason: The report mentions 'Patchy and linear opacities in the lung bases'

    2. Report: 'The lungs are clear without consolidation or edema.  There is no\n pleural effusion or pneumothorax.  There is mild cardiomegaly.  The\n mediastinal contours are normal.  The vertebral body heights are maintained in\n the thoracic spine.  No rib fractures identified.'
    Output:
    ['Cardiomegaly']
    Reason: The report mentions 'There is mild cardiomegaly'

    3. Report: 'The lungs are clear without focal consolidation.  No pleural effusion or\n pneumothorax is seen. The cardiac and mediastinal silhouettes are stable.\n Hilar contours are stable. There is persistent elevation of the right\n hemidiaphragm with minor right basilar atelectasis.'
    Output:
    ['Atelectasis']
    Reason: The report mentions 'There is persistent elevation of the right\n hemidiaphragm with minor right basilar atelectasis.'

    **********Report to Analyze. After output the list in the first line, provide your reasoning starting second line.:************
    {note}

    Output:
    """


def format_prompt(note):
    return f"""You are given a clinical report. Your task is to identify any diseases mentioned in the report and format the answer as a list.
Instructions:
1. Only use diseases from this list:
    ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
2. If there is no diease, return empty list. 
3. First thinks about the reasoning process in the mind and then provides the answer. 

############# Analyze the following report:
    {note}
############# Respond in the following format:
    <reasoning>...</reasoning>
    <answer>...</answer>
"""


def format_prompt2(note):
    return f"""You are given a clinical report. Your task is to identify any diseases mentioned in the report and format the answer as a list.
Instructions:
1. Only use diseases from this list:
    ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
2. If there is no diease, return empty list. 
3. First thinks about the reasoning process in the mind and then provides the answer. The reasoning should be based on phrases or evidence from the report.

############# Analyze the following report:
    {note}
"""


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

def format_prompt4(note):
    return f"""You are given a clinical report. Your task is to identify any diseases mentioned in the report and format the answer as a list.
Instructions:
1. Only use diseases from this list:
    ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
2. If there is no diease, return empty list. 
3. First thinks about the reasoning process in the mind and then provides the answer. The reasoning should be based on phrases or evidence from the report.

############# Analyze the following report:
    {note}
"""


def extract_think_answer(response):
    print("~~~~~~~~~~~~~~~~~")
    think_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

    think = think_match.group(1).strip() if think_match else None
    answer = answer_match.group(1).strip() if answer_match else None
    print("EXTRACTED ANSWER AND THINK !!!!!!!!!!!!!")
    print(answer)
    print(think)
    return answer, think


def structured_list(note):
    return f"""You are given a clinical report. Your task is to identify any diseases mentioned in the report and format the answer as a list.
Instructions:
1. Only use diseases from this list:
    ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
2. If there is no diease, return empty list. 
3. Output in a structured format, with two fields disease_list and reasoning. The reasoning should be based on phrases or evidence from the report.

############# Example ####################
## Report:
    Endotracheal tube terminates 6.9 cm above the carina.  The right subclavian
    line tip is at the mid SVC. The NG tube passes below the diaphragm and out of
    view.
## Answer:
    {{
        "disease_list": [Support Devices],
        "reasoning": Support devices is found because the report mentions: 'Endotracheal tube', 'subclavian line' and 'NG tube'.
    }}

############# Analyze the following report:
    {note}
"""


def structured_binary(note, disease):
    return f"""You are given a clinical report. Your task is to identify whether the report has the disease {disease}

    ### Instruction
    * Only diseases that are certain are considered. Phrases like "likely", "probable" do not indicate disease.
    * Output Yes or No in the first field
    * Give your reasoning in the second field. The reasoning should be based on phrases from the original report.

    ############# Analyze the following report:
    {note}
"""


def structured_binary_support_device(note, disease):
    return f"""You are given a clinical report. Your task is to identify whether the report has support devices.
    A support device in medicine refers to any medical tool or apparatus that is used to maintain, assist, or replace 
    a physiological function or provide structural support to a patient.

    ############# Instruction
    * Output Yes or No in the first field
    * Give your reasoning in the second field. The reasoning should be based on phrases from the original report.

    ############# Example ####################
    ## Report:
        Endotracheal tube terminates 6.9 cm above the carina.  The right subclavian
        line tip is at the mid SVC. The NG tube passes below the diaphragm and out of
        view.
    ## Answer:
        {{
            "answer": Yes,
            "reasoning": Support devices is found because the report mentions: 'Endotracheal tube', 'subclavian line' and 'NG tube'.
        }}

    ############# Analyze the following report:
    {note}
"""