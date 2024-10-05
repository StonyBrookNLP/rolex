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
        if(len(ovcs[i])==0):
            continue
        d = calc_single_sentence_construct_score(ovcs[i],y_pred_list[i],all_ovcs)
        # print(i,d)
        prec_list.append(d["Precision"])
        rec_list.append(d["Recall"])
        fn+=d["FN"]
    
    prec = sum(prec_list)/len(prec_list)*100
    rec = sum(rec_list)/len(rec_list)*100
    f1 = (2*prec*rec)/(prec+rec+1e-20)

    print(fn)
    

    return prec,rec,f1


def extract_ovcs(pk_list):
    ovcs = []
    # print("Start")
    for i in range(0,len(pk_list)):
        temp_list = pk_list[i].strip()
        # print(temp_list)
        if(temp_list==""):
            ovcs.append(pk_list[i])
        else:
            temp = temp_list.split("<SEP>")
            # print(temp)
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
        "results_incontext/chatgpt.csv",
        "results_incontext/chatgpt_parking.csv",
        "results_incontext/chatgpt_parking_bm25.csv",
        "results_incontext/gpt4.csv",
        "results_incontext/gpt4_parking.csv",
        "results_incontext/gpt4_parking_bm25.csv"
    ]
    
    file_to_score_df = files[4]
    # ovcs_df = f"dataset/nfs_rfc_pk.csv"
    ovcs_df = "dataset/nfs_rfc_NAACL.csv"
    
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
    y_pred = [i[i.find("A"):] for i in y_pred]
    y_actual = df_test["Actual Text"].tolist()
    y_gold_ovcs = extract_ovcs(df_ovc["pk_gold"].tolist())
    # print(y_gold_ovcs)

    x = sacrebleu.compute(predictions=y_pred, references=y_actual)["score"]
    y = rouge.compute(predictions=y_pred, references=y_actual)['rougeL']*100
    print("SacreBLEU & ROUGE")
    print(f"{x:.2f} & {y:.2f}")

    prec,rec,f1 = calc_construct_score(y_gold_ovcs,y_pred,all_ovcs)
    print("Construct scores")
    print(f"{prec:.2f} & {rec:.2f} & {f1:.2f}")

    print("Full Answer:")
    print(f"{x:.2f} & {prec:.2f} & {rec:.2f} & {f1:.2f}")


