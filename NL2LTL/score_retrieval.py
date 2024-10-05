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
    gold = [i.split("<SEP>") for i in gold]

    # retrieved = df["pk_retrieved"].tolist()
    retrieved = df["parking"].tolist()
    retrieved = [i.split("<SEP>") for i in retrieved]

    hit = 0
    total = 0
    

    for idx,array in enumerate(gold):
        for value in array:
            if(value==""):
                continue
            total+=1
            z = value.strip()
            # print(idx)
            # print(value)
            # print(retrieved[idx])
            if z in retrieved[idx]:
                # print("HIT!")
                hit+=1
            # print()
    
    print(f"Hit:{hit},Total:{total}")
    print(f"Recall:{hit/total*100:.2f}")