import pandas as pd
import ast
from collections import Counter

DISEASES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity',	'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
DISEASES = [d.lower() for d in DISEASES]

def get_invalid_list(diseases):
    result = []
    try:
        result = [d for d in diseases if d not in DISEASES and len(d) > 2]
    except:
        print("!!!!!!!!!!")
        print(diseases)
    return result        


def process_diseases(input_csv, output_csv):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)
    # df = df.iloc[-1:]

    def to_list(s, index):
        try:
            # Find the position of the first '['
            first_open_bracket = s.find('[')
            if first_open_bracket == -1:
                first_open_bracket = -1

            # Find the position of the first ']'
            first_close_bracket = s.find(']')
            if first_close_bracket == -1:
                first_close_bracket = len(s)

            # Extract the substring between the brackets
            substring = s[first_open_bracket + 1:first_close_bracket]
            substring = substring.split(",")
            substring = [x.strip(" ").strip("'") for x in substring]
            substring = [x.lower() for x in substring if len(x) > 2]
            return substring
        except Exception as e:
            print(index)
            print(s)
            print(e)
            return []

    def len_intersection(list1, list2):
        return len(set(list1) & set(list2))

    def majority_vote(row):
        all_items = row['generated_disease_list_0'] + row['generated_disease_list_1'] + row['generated_disease_list_2'] + row['generated_disease_list_3'] + row['generated_disease_list_4']
        # Count appearances
        counts = Counter(all_items)
        # Keep items that appear in at least 2 of the lists
        return [item for item, count in counts.items() if count >= 3]

    def majority_vote2(row):
        all_items = row['generated_disease_list_0'] + row['generated_disease_list_1'] + row['generated_disease_list_2'] + row['generated_disease_list_3'] + row['generated_disease_list_4']
        # Count appearances
        counts = Counter(all_items)
        # Keep items that appear in at least 2 of the lists
        return [item for item, count in counts.items() if count >= 2]

    # Convert the string representations to lists
    df['true_disease_list'] = df.apply(lambda row: to_list(row['true_answer'], row.name), axis=1)

    # df['generated_disease'] = df['generated_disease_0'].astype(str).str.replace('.', ' ', regex=False)
    # df['generated_disease_list'] = df.apply(lambda row: to_list(row['generated_disease'], row.name), axis=1)
    df['generated_disease_0'] = df['generated_disease_0'].astype(str).str.replace('.', ' ', regex=False)
    df['generated_disease_list_0'] = df.apply(lambda row: to_list(row['generated_disease_0'], row.name), axis=1)
    df['generated_disease_1'] = df['generated_disease_1'].astype(str).str.replace('.', ' ', regex=False)
    df['generated_disease_list_1'] = df.apply(lambda row: to_list(row['generated_disease_1'], row.name), axis=1)
    df['generated_disease_2'] = df['generated_disease_2'].astype(str).str.replace('.', ' ', regex=False)
    df['generated_disease_list_2'] = df.apply(lambda row: to_list(row['generated_disease_2'], row.name), axis=1)
    df['generated_disease_3'] = df['generated_disease_3'].astype(str).str.replace('.', ' ', regex=False)
    df['generated_disease_list_3'] = df.apply(lambda row: to_list(row['generated_disease_3'], row.name), axis=1)
    df['generated_disease_4'] = df['generated_disease_4'].astype(str).str.replace('.', ' ', regex=False)
    df['generated_disease_list_4'] = df.apply(lambda row: to_list(row['generated_disease_4'], row.name), axis=1)
    df['generated_disease_list'] = df.apply(majority_vote, axis=1)
    df['generated_disease_list_m2'] = df.apply(majority_vote2, axis=1)

    # df['generated_disease_invalid'] = df['generated_disease_list'].apply(get_invalid_list)

    # Calculate the lengths and intersections
    df['len_true_disease_list'] = df['true_disease_list'].apply(len)
    df['len_generated_disease_list_0'] = df['generated_disease_list_0'].apply(len)
    df['len_intersection_0'] = df.apply(lambda row: len_intersection(row['generated_disease_list_0'], row['true_disease_list']), axis=1)

    df['len_generated_disease_list_1'] = df['generated_disease_list_1'].apply(len)
    df['len_intersection_1'] = df.apply(lambda row: len_intersection(row['generated_disease_list_1'], row['true_disease_list']), axis=1)

    df['len_generated_disease_list_2'] = df['generated_disease_list_2'].apply(len)
    df['len_intersection_2'] = df.apply(lambda row: len_intersection(row['generated_disease_list_2'], row['true_disease_list']), axis=1)

    df['len_generated_disease_list_3'] = df['generated_disease_list_3'].apply(len)
    df['len_intersection_3'] = df.apply(lambda row: len_intersection(row['generated_disease_list_3'], row['true_disease_list']), axis=1)

    df['len_generated_disease_list_4'] = df['generated_disease_list_4'].apply(len)
    df['len_intersection_4'] = df.apply(lambda row: len_intersection(row['generated_disease_list_4'], row['true_disease_list']), axis=1)

    df['len_generated_disease_list'] = df['generated_disease_list'].apply(len)
    df['len_intersection'] = df.apply(lambda row: len_intersection(row['generated_disease_list'], row['true_disease_list']), axis=1)

    df['len_generated_disease_list_m2'] = df['generated_disease_list_m2'].apply(len)
    df['len_intersection_m2'] = df.apply(lambda row: len_intersection(row['generated_disease_list_m2'], row['true_disease_list']), axis=1)


    # df['len_invalid'] = df['generated_disease_invalid'].apply(len)
    # df['generated_but_wrong'] = df.apply(lambda row: list(set(row['generated_disease_list']) - set(row['true_disease_list'])), axis=1)
    # df['fail_to_generate'] = df.apply(lambda row: list(set(row['true_disease_list']) - set(row['generated_disease_list'])), axis=1)
    # df['success_generate'] = df.apply(lambda row: list(set(row['true_disease_list']) & set(row['generated_disease_list'])), axis=1)
    df = df[df['len_generated_disease_list'] < 10]
    # Drop the intermediate list columns
    df.drop(columns=['generated_disease_list_0', 'generated_disease_list_1', 'generated_disease_list_2', 'generated_disease_list_3', 'generated_disease_list_4'], inplace=True)

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)

input_csv = '/prj0129/yiw4018/reasoning/final/result/silver_data/qwen/sft_grpo.csv'
output_csv = input_csv
process_diseases(input_csv, output_csv)
 
