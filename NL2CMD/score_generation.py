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
    return {"Precision":precision,"Recall":recall,"F1-score":f1,"FN":false_negatives}

def convert_to_constructs(sentence,ovc_list):
    tokens = sentence.split(" ")
    constructs = []
    for j in tokens:
        if j in ovc_list:
            constructs.append(j)
    return constructs

def calc_single_sentence_construct_score(ovc,y_pred,all_ovcs):
    y_actual_constructs = ovc.split()
    y_pred_constructs = convert_to_constructs(y_pred,all_ovcs)

    return calculate_macro_metrics(y_actual_constructs,y_pred_constructs)

def calc_construct_score(ovcs,y_pred_list,all_ovcs):
    prec_list = []
    rec_list = []
    fn = 0

    for i in range(0,len(ovcs)):
        if(len(ovcs[i])==0):
            continue
        d = calc_single_sentence_construct_score(ovcs[i],y_pred_list[i],all_ovcs)
        # print(i,d)
        prec_list.append(d["Precision"])
        rec_list.append(d["Recall"])
        fn+=d["FN"]
    
    print("FN",fn)
    prec = sum(prec_list)/len(prec_list)*100
    rec = sum(rec_list)/len(rec_list)*100
    f1 = (2*prec*rec)/(prec+rec+1e-20)
    

    return prec,rec,f1


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

    # files = [
    #     "results/baseline_t5_small.csv",
    #     "results/parking_t5_small_supervision.csv",
    #     "results/baseline_t5_base.csv",
    #     "results/parking_t5_base_supervision.csv",
    #     "results/baseline_codet5_small.csv",
    #     "results/parking_codet5_small_supervision.csv",
    #     "results/baseline_codet5_base.csv",
    #     "results/parking_codet5_base_supervision.csv",
    # ]

    files = [
        "results/baseline_t5_small.csv",
        "results/parking_t5_small.csv",
        "results/baseline_t5_base.csv",
        "results/parking_t5_base.csv",
        "results/baseline_t5_large.csv",
        "results/parking_t5_large.csv",

        "results/baseline_codet5_small.csv",
        "results/parking_codet5_small.csv",
        "results/baseline_codet5_base.csv",
        "results/parking_codet5_base.csv",
        "results/baseline_codet5_large.csv",
        "results/parking_codet5_large.csv",

        "results/parking_codet5_base.csv",
        "results/parking_codet5_base_supervision.csv",
        "results/parking_codet5_base_basic.csv",

        "results/baseline_t5_base_more.csv",
    ]

    file_to_score_df = files[9]
    print(file_to_score_df)
    #file_to_score_df = "models/parking_plbart_base/predictions_2.csv"
    #file_to_score_df = "results/a2.csv"


    df_test = pd.read_csv(file_to_score_df)
    df_test.fillna("",inplace=True)

    sacrebleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")

    #Get all the possible OVCs
    df_ovcs = pd.read_csv("dataset/pk.csv")
    all_ovcs = df_ovcs["fl"].tolist()
    all_ovcs = " ".join(all_ovcs)

    y_pred = df_test["Generated Text"].tolist()
    y_actual = df_test["Actual Text"].tolist()
    y_gold_ovcs = extract_ovcs(df_test["Gold_PK"])
    if("parking" in file_to_score_df):
        y_adjusted_ovcs = extract_ovcs(df_test["Adjustment"].tolist())

    x = sacrebleu.compute(predictions=y_pred, references=y_actual)["score"]
    y = rouge.compute(predictions=y_pred, references=y_actual)['rougeL']*100
    print("SacreBLEU & ROUGE")
    print(f"{x:.2f} & {y:.2f}")

    prec,rec,f1 = calc_construct_score(y_gold_ovcs,y_pred,all_ovcs)
    print("Construct scores")
    print(f"{prec:.2f} & {rec:.2f} & {f1:.2f}")

    if("parking" in file_to_score_df):
        prec_adjusted,rec_adjusted,f1_adjusted = calc_construct_score(y_adjusted_ovcs,y_pred,all_ovcs)
        print("Adjusted Construct scores")
        print(f"{prec_adjusted:.2f} & {rec_adjusted:.2f} & {f1_adjusted:.2f}")

    print("Full Answer:")
    print(f"{x:.2f} & {prec:.2f} & {rec:.2f} & {f1:.2f}")
    if ("parking" in file_to_score_df):
        print(f"{rec_adjusted:.2f}")