import numpy as np
import pandas as pd
import random
import json

if __name__ == "__main__":
    input_file = "dataset/train_augmented.csv"
    output_file = "dataset/train_augmented.json"

    #Processing the retriever to json
    df = pd.read_csv(input_file)
    df.fillna("",inplace=True)


    #Create and save the json
    question_list = df["nl"]
    context_list = df["pk_gold"]
    context_list = [i.split("<SEP>") for i in context_list]

    json_object = []

    for i in range(0,len(context_list)):
        d = {}
        d["query"] = question_list[i]
        context = [x.split("=>")[0] for x in context_list[i]]
        d["pos"] = context
        d["neg"] = []
        json_object.append(d)

    # Serializing json
    json_object = json.dumps(json_object, indent=3)
    
    # Writing to sample.json
    with open(output_file, "w") as outfile:
        outfile.write(json_object)
