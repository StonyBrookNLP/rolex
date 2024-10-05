import pandas as pd
import numpy as np
import random
import argparse


if __name__ == "__main__":
    file_to_process = "dataset/dummy_train.csv"

    df = pd.read_csv(file_to_process)
    df.fillna("",inplace=True)
    nl = df["translation"].tolist()
    gold = df["lexicon"].tolist()
    gold = [i.split("$$") for i in gold]

    retrieved = df["retrieved_pk"].tolist()
    retrieved = [i.split("$$$") for i in retrieved]

    gold_among_retrieved = []
    

    for idx,array in enumerate(gold):
        temp = []
        for value in array:
            z = value.strip()
            if z in retrieved[idx]:
                temp.append(value)
        gold_among_retrieved.append("$$$".join(temp))

    df["gold_among_retrieved"] = gold_among_retrieved

    df.to_csv("dataset/dummy_train.csv")