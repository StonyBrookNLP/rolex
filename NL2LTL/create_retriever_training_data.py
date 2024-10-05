import numpy as np
import pandas as pd
import random

if __name__ == "__main__":
    train_df = pd.read_csv("dataset/RAG_train.csv")#[0:40000]
    train_df.fillna("",inplace=True)
    # train_df.dropna(inplace=True)
    input1 = train_df["nl"].tolist()
    input2 = train_df["pk_gold"].tolist()

    final_1 = []
    final_2 = []

    for i in range(0,len(input1)):
        temp_input2 = input2[i].split("<SEP>")
        for j in temp_input2:
            final_1.append(input1[i])
            final_2.append(j.split("=>")[0].strip())

    new_df = pd.DataFrame({"sentence_1":final_1,"sentence_2":final_2})
    new_df.to_csv("dataset/retriever_train.csv")