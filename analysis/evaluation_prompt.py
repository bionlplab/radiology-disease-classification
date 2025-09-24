vllm_schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://example.com/medical_phrase_extraction.schema.json",
    "title": "MedicalPhraseExtraction",
    "description": "Schema for extracting medical phrases and their attributes from clinical reports.",
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "phrase": {
                        "type": "string",
                        "description": "Extracted phrase relevant to medical findings."
                    },
                    "whether_supported_by_report": {
                        "type": "boolean",
                        "description": "Indicates if the phrase is supported by the report content."
                    },
                    "target_disease": {
                        "type": "string",
                        "description": "The diseases or condition associated with the phrase."
                    }
                },
                "required": ["phrase", "whether_supported_by_report", "target_disease"],
                "additionalProperties": False
            },
            "description": "List of extracted medical phrases with relevant attributes."
        }
    },
    "required": ["results"],
    "additionalProperties": False
}



output_schema = {
    "type": "json_schema",
    "json_schema": {
      "name": "medical_phrase_extraction",
      "strict": True,
      "schema": {
        "type": "object",
        "properties": {
          "results": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "phrase": {
                  "type": "string",
                  "description": "Extracted phrase relevant to medical findings."
                },
                "whether_supported_by_report": {
                  "type": "boolean",
                  "description": "Indicates if the phrase is supported by the report content."
                },
                "target_disease": {
                  "type": "string",
                  "description": "The disease or condition associated with the phrase."
                }
              },
              "required": ["phrase", "whether_supported_by_report", "target_disease"],
              "additionalProperties": False
            }
          }
        },
        "required": ["results"],
        "additionalProperties": False
      }
    }
  }



def gpt_evaluation_prompt(note, reasoning, result):
    return f"""
    There is an AI assistant that tend to extract diseases from radiology report. You are given the report, the reasoning
    given by the assistant and the result given by the assistant. Your task is to evaluate whether the AI assistant is doing
    a correct job.

    ********* Instructions:
    1. The output is a list of formatted structures, each element in the list has the components of: 'phrase', 
    'whether supported by report', 'target diseases', 'whether lead to the answer'
    2. For the 'phrase' component, your task is to divide the reasoning into semantically independent part. Each part will lead to 
    one structured element returned, and the part will be the 'phrase' component.
    3. 'whether_supported_by_report' is a boolean to evaluate whether the 'phrase' is supported by the report.
    4. 'target_diseases': which diseases are targeted by this phrase. Here all diseases come in this list:
        ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
        Each phrase can target to a several diseases or even no disease. So 'target diseases' are a list of candidate disease this 
        'phrase' is targeting for
    5. If there is no reasoning or the reasoning is clearly just repeating a list or not making sense, return empty list

    ********** Example
        ## Report
            FINDINGS: A cluster of heterogeneous opacities in the right lower lung has 
            has continued to grow since ___. 
            Otherwise, the lungs are clear. Moderate cardiomegaly, including severe left
            atrial enlargement is chronic; there is no pulmonary vascular congestion or
            edema. The thoracic aorta is heavily calcified.  There may be a new small,
            right pleural effusions or pneumothorax.
            IMPRESSION: Slowly progressive chronic right pneumonia, could be exogenous
            lipoid pneumonia, but tuberculosis is in the differential.  CT scanning
            recommended.  Nurse ___ and I discussed the findings and their
            clinical significance by telephone at the time of dictation.
        ## Reasoning
            According to the report, there is a cluster of heterogeneous opacities in the right lower lung.
            A pneumonia has developed.  There are bilateral pleural effusions, one on the right and one on the left.
            Additionally, there may be a small pneumothorax at the right lung base.  Supportive devices are mentioned, 
            like a chest tube, but not specified.
        ## Result
            ['Pneumonia', 'Pleural Effusion', 'Pneumothorax', 'Support Devices']   
        
        ###### Your output should be:
        [{{
            'phrase': 'According to the report, there is a cluster of heterogeneous opacities in the right lower lung.',
            'whether_supported_by_report': True,
            'target_diseases': ['Lung Opacity'],
        }},
        {{
            'phrase': 'A pneumonia has developed',
            'whether_supported_by_report': True,  (The report mentions 'Slowly progressive chronic right pneumonia')
            'target_diseases': ['Pneumonia'],
        }},
        {{
            'phrase': 'There are bilateral pleural effusions, one on the right and one on the left.',
            'whether_supported_by_report': False,  (The report mentions it is on the right)
            'target_diseases': ['Pleural effusion'],
        }},
        {{
            'phrase': 'Additionally, there may be a small pneumothorax at the right lung base.', 
            'whether_supported_by_report': True,  (Although the report mentions on the right, not right base, but it is very close, so mark as True)
            'target_diseases': ['Pneumothorax'],
        }},
        {{
            'phrase': 'Supportive devices are mentioned, like a chest tube, but not specified.',
            'whether_supported_by_report': False,  
            'target_diseases': ['Support Devices'],
        }},
        ]

    ********** Here is your input
        ## Report
            {note}
        ## Reasoning
            {reasoning}
        ## Result
            {result}
    ********** Give your output
    """