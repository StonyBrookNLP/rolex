import pandas as pd
import numpy as np
import random
import argparse


if __name__ == "__main__":
    random.seed(42)
    OPEN_FILE = "dataset/train_augmented_retrieved.csv"
    SAVE_FILE = "dataset/train_generator.csv"

    # OPEN_FILE = "dataset/val_augmented_retrieved.csv"
    # SAVE_FILE = "dataset/val_generator.csv"

    df = pd.read_csv(OPEN_FILE)
    df.fillna("",inplace=True)
    nl = df["nl"].tolist()
    gold = df["pk_gold"].tolist()
    gold = [i.split("<SEP>") for i in gold]

    retrieved = df["pk_retrieved"].tolist()
    retrieved = [i.split("<SEP>") for i in retrieved]

    full_recall = []
    weak_target = []
    weak_pk = []

    for idx,array in enumerate(gold):
        temp_gold = []
        temp_weak = []
        temp_weak_pk = []
        temp_retrieved = retrieved[idx].copy()
        for value in array:
            if(value==""):
                continue
            z = value.strip()
            temp_gold.append(z)
            if z in temp_retrieved:
                temp_weak.append(z.split("=>")[1])
                temp_weak_pk.append(z)
                temp_retrieved.remove(z)
        temp = temp_retrieved+temp_gold
        random.shuffle(temp)
        full_recall.append("<SEP>".join(temp))
        if(len(temp_weak)==0):
            temp_weak=["NONE"]
        weak_target.append("<SEP>".join(temp_weak))
        weak_pk.append("<SEP>".join(temp_weak_pk))

    df["full_recall"] = full_recall
    df["pk_generator"] = df["pk_retrieved"]
    df["weak_target"] = weak_target
    df["weak_pk"] = weak_pk

    df.to_csv(SAVE_FILE)