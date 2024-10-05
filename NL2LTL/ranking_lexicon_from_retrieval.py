import pandas as pd
import numpy as np
import random
import argparse


if __name__ == "__main__":
    file_to_process = "dataset/dummy_test.csv"

    df = pd.read_csv(file_to_process)
    df.fillna("",inplace=True)
    nl = df["translation"].tolist()
    gold = df["lexicon"].tolist()
    gold = [i.split("$$") for i in gold]

    retrieved = df["retrieved_pk"].tolist()
    retrieved = [i.split("$$$") for i in retrieved]

    ranking_retrieved = []
    

    for idx,array in enumerate(gold):
        temp = []
        for value in array:
            z = value.strip()
            if z in retrieved[idx]:
                temp.append(z)
        new_lex = retrieved[idx].copy()
        for i in temp:
            if i in new_lex:
                new_lex.remove(i)
        new_lex = temp+new_lex
        ranking_retrieved.append("$$$".join(new_lex))

    df["ranking_retrieved"] = ranking_retrieved

    df.to_csv("dataset/dummy_rank_test.csv")