import json


def get_list(input):
    try:
        start = input.find("[")
        end = input.find("]")
        if end == -1:
             end = len(input)
        if len(input[start + 1:end].strip()) == 0:
            return []

        input = input[start + 1:end].split(",")

        input = [x.strip().strip('"').strip("'").lower() for x in input]
        return input
    except Exception as e:
         print(e)
         print(input)
         return []

def analyze_json_output(evaluation, generated_disease, true_disease):
    try:
        evaluation = json.loads(evaluation)
        results = evaluation["results"]
        total_count = len(results)
        supported_count = sum(1 for item in results if item["whether_supported_by_report"])

        # Aggregate target_disease
        target_diseases = list(set(item["target_disease"] for item in results))
        target_diseases = [t.lower() for t in target_diseases if t != '']
        if not isinstance(generated_disease, list):
            generated_disease = get_list(generated_disease)
        if not isinstance(true_disease, list):
            true_disease = get_list(true_disease)
        return {
            "target_diseases": target_diseases,
            "generated_disease": generated_disease,
            "true_disease": true_disease,
            "total_reason_count": total_count,
            "correct_reason_count": supported_count,
            "total_target_disease": len(target_diseases),
            "missed_diseases": len(set(true_disease) - set(target_diseases)),
            "no_support_generation": len(set(generated_disease) - set(target_diseases)),
            "len_generation": len(generated_disease),
            "len_target": len(target_diseases),
            "len_true": len(true_disease),
            # appears in target, appears in generation, appears in true / appears in target, not appears in generation, not appears in true
            "inference_correct": len(set(target_diseases)&set(generated_disease)&set(true_disease))+len((set(target_diseases)-set(generated_disease))&(set(target_diseases)-set(true_disease)))
        }
    except Exception as e:
        print(e)
        print(evaluation)
        return None
