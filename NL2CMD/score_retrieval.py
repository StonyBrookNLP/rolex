import pandas as pd
import numpy as np
import random
import argparse


if __name__ == "__main__":
    file_to_process = "retriever/bge_large_ft_20.csv"

    df = pd.read_csv(file_to_process)
    df.fillna("",inplace=True)
    nl = df["nl"].tolist()
    gold = df["pk_gold"].tolist()
    # gold = [i.split("|") for i in gold]

    #retrieved = df["pk_retrieved"].tolist()
    retrieved = df["parking"].tolist()
    retrieved = [i.split("<SEP>") for i in retrieved]

    hit = 0
    total = 0

    # print("GOLD")
    # print(gold)
    # print("END GOLD")

    for idx,value in enumerate(gold):
        # print(idx)
        # print(value)
        # print(retrieved[idx])
        total+=1
        if value in retrieved[idx]:
            hit+=1
    
    print(f"Hit:{hit},Total:{total}")
    print(f"Recall:{hit/total*100:.2f}")