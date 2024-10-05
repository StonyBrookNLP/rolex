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
    ovc_constructs = ovc_list.split("<SEP>")
    # print(tokens)
    for j in ovc_constructs:
        if j in sentence:
            # print(j," | ",sentence)
            constructs.append(j)
    return constructs

def calc_single_sentence_construct_score(ovc,y_pred,all_ovcs):
    y_actual_constructs = ovc.split("<SEP>")
    y_pred_constructs = convert_to_constructs(y_pred,all_ovcs)

    # print(y_actual_constructs)
    # print(ovc)
    # print(y_pred_constructs)
    # print(y_pred)
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
    print("FN:",fn)
    prec = sum(prec_list)/len(prec_list)*100
    rec = sum(rec_list)/len(rec_list)*100
    f1 = (2*prec*rec)/(prec+rec+1e-20)
    

    return prec,rec,f1


# def calc_knowledge_recall(y_pk_at_t_list,y_pred_list,ovc_list):
#     rec_list = []
#     rec_dict = {}

#     for i in range(0,len(y_pk_at_t_list)):
#         fls = y_pk_at_t_list[i]
#         if(len(fls)==0):
#             continue
#         fls = fls.split("<SEP>")
#         fls = [i.split("=>")[1] for i in fls]
#         fls = " ".join(fls)

#         d,y_pred_construct,y_actual_construct = calc_single_sentence_construct_score(fls,y_pred_list[i],ovc_list)
#         rec_list.append(d["Recall"])
#         rec_dict[i] = [y_pred_construct,y_actual_construct,d["Recall"]]
    
#     rec = sum(rec_list)/len(rec_list)*100
#     return rec,rec_dict


def extract_ovcs(pk_list):
    ovcs = []
    for i in range(0,len(pk_list)):
        temp_list = pk_list[i].strip()
        if(temp_list==""):
            ovcs.append(pk_list[i])
        else:
            temp = temp_list.split("<SEP>")
            temp = [x.split("=>")[1] for x in temp]

            for j in range(0,len(temp)):
                if "(" in temp[j]:
                    temp[j] = temp[j][:temp[j].find("(")+1]
                temp[j] = temp[j].strip()
            temp = "<SEP>".join(temp)
            ovcs.append(temp)
    return ovcs


if __name__ == "__main__":

    files = [
        # "results_nfs_rfc/baseline_t5_small.csv",
        # "results_nfs_rfc/parking_t5_small.csv",
        # "results_nfs_rfc/baseline_t5_base.csv",
        # "results_nfs_rfc/parking_t5_base.csv",
        # "results_nfs_rfc/baseline_t5_large.csv",
        # "results_nfs_rfc/parking_t5_large.csv",
        # "results_nfs_rfc/baseline_codet5_small.csv",
        # "results_nfs_rfc/parking_codet5_small.csv",
        # "results_nfs_rfc/baseline_codet5_base.csv",
        # "results_nfs_rfc/parking_codet5_base.csv",
        # "results_nfs_rfc/parking_t5_small_basic.csv",
        # "results_nfs_rfc/baseline_codet5_large.csv",
        # "results_nfs_rfc/parking_codet5_large.csv",
        # "results_nfs_rfc/parking_codet5_base_uu.csv",

        "results_nfs_rfc_more/baseline_t5_small.csv",
        "results_nfs_rfc_more/parking_t5_small.csv",
        "results_nfs_rfc_more/baseline_t5_base.csv",
        "results_nfs_rfc_more/parking_t5_base.csv",
        "results_nfs_rfc_more/baseline_codet5_small.csv",
        "results_nfs_rfc_more/parking_codet5_small.csv",
        "results_nfs_rfc_more/baseline_codet5_base.csv",
        "results_nfs_rfc_more/parking_codet5_base.csv"

    ]
    
    file_to_score_df = files[7]
    #file_to_score_df = "results_nfs_rfc/parking_codet5_base_gtr_large.csv"
    # ovcs_df = f"dataset/nfs_rfc_pk.csv"
    ovcs_df = "dataset/test_nfs_rfc.csv"
    
    df_test = pd.read_csv(file_to_score_df)
    df_test.fillna("",inplace=True)

    df_ovc = pd.read_csv(ovcs_df)
    df_ovc.fillna("",inplace=True)

    #Get all the possible OVCs
    df_ovcs = pd.read_csv("dataset/nfs_rfc_pk.csv")
    all_ovcs = extract_ovcs(df_ovcs["pk"].tolist())
    all_ovcs = set(all_ovcs)
    all_ovcs = "<SEP>".join(all_ovcs)

    sacrebleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")

    y_pred = df_test["Generated Text"].tolist()
    y_actual = df_test["Actual Text"].tolist()
    y_gold_ovcs = extract_ovcs(df_ovc["pk_gold"].tolist())
    y_adjusted_ovcs = extract_ovcs(df_ovc["knowledge_adjusted_gold"].tolist())

    x = sacrebleu.compute(predictions=y_pred, references=y_actual)["score"]
    y = rouge.compute(predictions=y_pred, references=y_actual)['rougeL']*100
    print("SacreBLEU & ROUGE")
    print(f"{x:.2f} & {y:.2f}")

    prec,rec,f1 = calc_construct_score(y_gold_ovcs,y_pred,all_ovcs)
    print("Construct scores")
    print(f"{prec:.2f} & {rec:.2f} & {f1:.2f}")

    prec_adjusted,rec_adjusted,f1_adjusted = calc_construct_score(y_adjusted_ovcs,y_pred,all_ovcs)
    print("Adjusted Construct scores")
    print(f"{prec_adjusted:.2f} & {rec_adjusted:.2f} & {f1_adjusted:.2f}")


    print("Full Answer:")
    print(f"{x:.2f} & {prec:.2f} & {rec:.2f} & {f1:.2f}")


