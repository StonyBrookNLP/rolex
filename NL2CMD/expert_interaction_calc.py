import numpy as np
import pandas as pd
import evaluate
import ast


def calculate_error_number(ground_truth,predicted):
    true_positives = len(set(ground_truth) & set(predicted))
    return len(ground_truth) - true_positives

def convert_to_constructs(sentence,ovc_list):
    tokens = sentence.split(" ")
    constructs = []
    for j in tokens:
        if j in ovc_list:
            constructs.append(j)
    return constructs


def calc_single_sentence_error_rate(ovc,y_pred):
    y_actual_constructs = ovc.split()
    y_pred_constructs = convert_to_constructs(y_pred,ovc)

    return calculate_error_number(y_actual_constructs,y_pred_constructs)


def extract_ovcs(pk_list):
    ovcs = []
    for i in range(0,len(pk_list)):
        if(pk_list[i]==""):
            ovcs.append(pk_list[i])
        else:
            temp = pk_list[i].split("<SEP>")
            temp = [i.split("=>")[1] for i in temp]
            temp = " ".join(temp)
            ovcs.append(temp)
    return ovcs



if __name__ == "__main__":
    file_to_score = f"results/parking_codet5_base.csv"

    #Check if it is a parking model
    is_parking = False
    if "parking" in file_to_score:
        is_parking = True
    
    ovcs_df = f"dataset/test_parking.csv"
    
    df_test = pd.read_csv(file_to_score)
    df_test.fillna("",inplace=True)

    sacrebleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")

    y_pred = df_test["Generated Text"].tolist()
    y_actual = df_test["Actual Text"].tolist()
    y_gold_ovcs = extract_ovcs(df_test["Gold_PK"])
    y_parking = df_test["Gold_PK"]
    y_parking = [i.split("<SEP>") for i in y_parking]

    reading_cost = len(y_pred)
    error_cost = 0
    insert_lexicon_cost = 0
    pk_till_now = {}

    for i in range(0,len(y_pred)):
        error_cost+= calc_single_sentence_error_rate(y_gold_ovcs[i],y_pred[i])

        for j in y_parking[i]:
            if j not in pk_till_now:
                insert_lexicon_cost+=1
                pk_till_now[j]=True
    
    if is_parking:
        total_cost = reading_cost+error_cost
    else:
        total_cost = reading_cost+error_cost
    
    print(f"Model:{file_to_score}")
    print(f"Reading cost:{reading_cost}")
    print(f"Error cost:{error_cost}")
    print(f"Insertion cost:{insert_lexicon_cost}")
    print(f"Total cost:{total_cost}")