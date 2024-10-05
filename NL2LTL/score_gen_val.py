import numpy as np
import pandas as pd
import evaluate
import ast

def get_score(filename):
    df_test = pd.read_csv(filename)
    df_test.fillna("",inplace=True)

    sacrebleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")

    y_pred = df_test["Generated Text"].tolist()

    if "<SEP>" in y_pred[0]:
        y_pred = [i.split("<SEP>")[1] for i in y_pred]

    y_actual = df_test["Actual Text"].tolist()

    x = sacrebleu.compute(predictions=y_pred, references=y_actual)["score"]
    
    return x

if __name__ == "__main__":

    files = [
        "models/baseline_t5_small",
        "models/baseline_t5_base",
        "models/baseline_t5_large",
        "models/baseline_codet5_small",
        "models/baseline_codet5_base",
        "models/baseline_codet5_large",

        "models/parking_t5_small",
        "models/parking_t5_base",
        "models/parking_t5_large",
        "models/parking_codet5_small",
        "models/parking_codet5_base",
        "models/parking_codet5_large",

        
        "models_2/parking_codet5_base",
    ]

    idx = -1

    files_to_score = [f"{files[idx]}/predictions_{i}.csv" for i in range(0,1)]

    scores = [get_score(files_to_score[i]) for i in range(0,len(files_to_score))]


    print(scores)
    print("Best val:",scores.index(max(scores)))


    