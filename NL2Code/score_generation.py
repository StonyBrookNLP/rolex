import numpy as np
import pandas as pd
import evaluate
import ast
from score_generation_docprompt import calculate_bleu

def get_func_names_from_OVC(ovc_list):
    func_names = []
    for item in ovc_list:

        if "#" in item:
            idx = item.find("#")
            func_names.append(item[idx+1:])
        elif "." in item:
            idx = item.rfind(".")
            func_names.append(item[idx+1:])
        else:
            func_names.append(item)
    return func_names


def calculate_macro_metrics(ground_truth, predicted):
    true_positives = len(set(ground_truth) & set(predicted))
    false_positives = len(predicted) - true_positives
    false_negatives = len(ground_truth) - true_positives
    precision = true_positives / (true_positives + false_positives+1e-20)
    recall = true_positives / (true_positives + false_negatives+1e-20)
    f1 = (2*precision*recall)/(precision+recall+1e-20)
    return {"Precision":precision,"Recall":recall,"F1-score":f1,"FN":false_negatives}

def convert_to_constructs(sentence,ovc_list):
    ovcs = ovc_list.split()
    func_names = get_func_names_from_OVC(ovcs)
    constructs = []
    for j in func_names:
        if j in sentence:
            constructs.append(j)
    return constructs

def calc_single_sentence_construct_score(ovc,y_pred,all_ovcs):
    y_actual_constructs = get_func_names_from_OVC(ovc.split())
    y_pred_constructs = convert_to_constructs(y_pred,all_ovcs)
    # print(all_ovcs)
    # print(y_pred_constructs)

    # print(ovc)
    # print(y_pred)
    # print(y_actual_constructs)
    # print(y_pred_constructs)
    # print()

    return calculate_macro_metrics(y_actual_constructs,y_pred_constructs)

def calc_construct_score(ovcs,y_pred_list,all_ovcs):
    prec_list = []
    rec_list = []
    fn = 0

    for i in range(0,len(ovcs)):
        # print(i)
        if(len(ovcs[i])==0):
            continue
        d = calc_single_sentence_construct_score(ovcs[i],y_pred_list[i],all_ovcs)
        # print(i,d)
        prec_list.append(d["Precision"])
        rec_list.append(d["Recall"])
        fn+=d["FN"]
    
    # print(sum(prec_list))
    # print(sum(rec_list))
    prec = sum(prec_list)/(len(prec_list)+1e-20)
    rec = sum(rec_list)/(len(rec_list)+1e-20)
    f1 = (2*prec*rec)/(prec+rec+1e-20)
    print(fn)
    

    return prec*100,rec*100,f1*100

def extract_ovcs(pk_list):
    ovcs = []
    for i in range(0,len(pk_list)):
        if(pk_list[i]=="" or pk_list[i]==" "):
            ovcs.append(pk_list[i])
        else:
            temp = pk_list[i].split("<SEP>")
            temp = [i.split("=>")[1] for i in temp]
            temp = " ".join(temp)
            ovcs.append(temp)
    return ovcs


if __name__ == "__main__":

    files = [
        "results/baseline_t5_small.csv",
        "results/parking_t5_small.csv",

        "results_more/baseline_t5_base.csv",
        "results_more/parking_t5_base.csv",

        "results_more/baseline_t5_large.csv",
        "results_more/parking_t5_large.csv",

        "results_more/baseline_codet5_small.csv",
        "results_more/parking_codet5_small.csv",

        "results_more/baseline_codet5_base.csv",
        "results_more/parking_codet5_base.csv",

        "results_more/baseline_codet5_large.csv",
        "results_more/parking_codet5_large.csv",

        "results/pairs.csv",
        "results/docs.csv",

        # "results_more/parking_codet5_base.csv",
        # "results_more/parking_codet5_base_supervision.csv",
        # "results_more/parking_codet5_base_basic.csv",
        # "results_more/parking_codet5_base_together.csv",

        # "results/parking_bm25.csv",
        # "results/parking_bge_small_ut.csv",
        # "results/parking_bge_large_ut.csv",
        # "results/parking_bge_small_ft.csv",

        # "results_more/parking_codet5_small.csv"

    ]

    # file_to_score_df = "baseline_t5_base_sd.csv"
    file_to_score_df = files[3]
    #file_to_score_df = "models/baseline_t5_small_again/predictions_4.csv"
    # file_to_score_df = "results/a2.csv"

    print(file_to_score_df)


    df_test = pd.read_csv(file_to_score_df)
    df_test.fillna("",inplace=True)

    #Get all the possible OVCs
    df_ovcs = pd.read_csv("dataset/test_pk_new.csv")
    all_ovcs = df_ovcs["fl"].tolist()
    all_ovcs = " ".join(all_ovcs)

    sacrebleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")

    y_pred = df_test["Generated Text"].tolist()

    

    if "<SEP>" in y_pred[0]:
        y_pred = [i.split("<SEP>")[-1] for i in y_pred]
    
    print(y_pred[342])

    y_actual = df_test["Actual Text"].tolist()

    #Fix the GOLD PK issue with updated pk list
    gold_pk_list = df_test["Gold_PK"].tolist()
    
    actual_golds_set = pd.read_csv("dataset/test_pk_new.csv")["pk"].tolist()
    actual_golds_set = set(actual_golds_set)
    # print(actual_golds_set)

    new_gold_pk_list = []
    for idx,item in enumerate(gold_pk_list):
        x = item.split("<SEP>")
        temp = []
        for i in x:
            if i in actual_golds_set:
                temp.append(i)
        temp = "<SEP>".join(temp)
        new_gold_pk_list.append(temp)

    # print(new_gold_pk_list)

    y_gold_ovcs = extract_ovcs(new_gold_pk_list)

    # if("parking" in file_to_score_df):
    #     y_adjusted_ovcs = extract_ovcs(df_test["Adjustment"].tolist())

    # print(y_gold_ovcs)

    x = sacrebleu.compute(predictions=y_pred, references=y_actual)["score"]
    y = rouge.compute(predictions=y_pred, references=y_actual)['rougeL']*100
    print("SacreBLEU & ROUGE")
    print(f"{x:.2f} & {y:.2f}")

    prec,rec,f1 = calc_construct_score(y_gold_ovcs,y_pred,all_ovcs)
    print("Construct scores")
    print(f"{prec:.2f} & {rec:.2f} & {f1:.2f}")

    # if("parking" in file_to_score_df):
    #     prec_adjusted,rec_adjusted,f1_adjusted = calc_construct_score(y_adjusted_ovcs,y_pred)
    #     print("Adjusted Construct scores")
    #     print(f"{prec_adjusted:.2f} & {rec_adjusted:.2f} & {f1_adjusted:.2f}")
    docprompt_bleu = calculate_bleu(y_pred,y_actual)

    print("Full Answer:")
    print(f"{x:.2f} & {docprompt_bleu:.2f} & {prec:.2f} & {rec:.2f} & {f1:.2f}")