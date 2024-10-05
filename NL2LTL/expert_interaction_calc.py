import numpy as np
import pandas as pd
import evaluate



def calculate_macro_metrics(ground_truth, predicted):
    true_positives = len(set(ground_truth) & set(predicted))
    false_positives = len(predicted) - true_positives
    false_negatives = len(ground_truth) - true_positives
    precision = true_positives / (true_positives + false_positives+1e-20)
    recall = true_positives / (true_positives + false_negatives+1e-20)
    f1 = (2*precision*recall)/(precision+recall+1e-20)
    return {"Precision":precision,"Recall":recall,"F1-score":f1}

def calculate_error_number(ground_truth,predicted):
    true_positives = len(set(ground_truth) & set(predicted))
    return len(ground_truth) - true_positives

def convert_to_constructs(sentence,ovc_list):
    tokens = sentence.split(" ")
    constructs = []
    # print(tokens)
    for j in tokens:
        if j in ovc_list:
            constructs.append(j)
    return constructs

def calc_single_sentence_construct_score(ovc,y_pred):
    y_actual_constructs = ovc.split("<SEP>")
    y_pred_constructs = convert_to_constructs(y_pred,ovc)

    return calculate_macro_metrics(y_actual_constructs,y_pred_constructs)


def calc_single_sentence_error_rate(ovc,y_pred):
    y_actual_constructs = ovc.split("<SEP>")
    y_pred_constructs = convert_to_constructs(y_pred,ovc)

    return calculate_error_number(y_actual_constructs,y_pred_constructs)

def calc_construct_score(ovcs,y_pred_list):
    prec_list = []
    rec_list = []

    for i in range(0,len(ovcs)):
        if(len(ovcs[i])==0):
            continue
        d = calc_single_sentence_construct_score(ovcs[i],y_pred_list[i])
        # print(i,d)
        prec_list.append(d["Precision"])
        rec_list.append(d["Recall"])
    
    prec = sum(prec_list)/len(prec_list)*100
    rec = sum(rec_list)/len(rec_list)*100
    f1 = (2*prec*rec)/(prec+rec+1e-20)
    

    return prec,rec,f1


def extract_ovcs(pk_list):
    ovcs = []
    for i in range(0,len(pk_list)):
        temp_list = pk_list[i].strip()
        if(temp_list==""):
            ovcs.append(pk_list[i])
        else:
            temp = temp_list.split("<SEP>")
            temp = [i.split("=>")[1] for i in temp]

            for j in range(0,len(temp)):
                if "(" in temp[j]:
                    temp[j] = temp[j][:temp[j].find("(")]
                temp[j] = temp[j].strip()
            temp = "<SEP>".join(temp)
            ovcs.append(temp)
    return ovcs






if __name__ == "__main__":
    file_to_score = f"results_nfs_rfc/parking_t5_base.csv"

    #Check if it is a parking model
    is_parking = False
    if "parking" in file_to_score:
        is_parking = True
    
    ovcs_df = f"dataset/test_nfs_rfc.csv"
    
    df_test = pd.read_csv(file_to_score)
    df_test.fillna("",inplace=True)

    df_ovc = pd.read_csv(ovcs_df)
    df_ovc.fillna("",inplace=True)

    sacrebleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")

    y_pred = df_test["Generated Text"].tolist()
    y_actual = df_test["Actual Text"].tolist()
    y_gold_ovcs = extract_ovcs(df_ovc["pk_gold"].tolist())
    y_parking = df_ovc["pk_gold"]
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