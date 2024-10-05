import numpy as np
import random
import pandas as pd
from datasets import load_dataset

if __name__ == "__main__" :
    # Load the dataset
    tldr_data = load_dataset("neulab/tldr",name="data")
    tldr_docs = load_dataset("neulab/tldr",name="docs")
    train_df = pd.DataFrame(tldr_data["train"])
    val_df = pd.DataFrame(tldr_data["validation"])
    test_df = pd.DataFrame(tldr_data["test"])
    schema_df = pd.DataFrame(tldr_docs["train"])

    #Save to csv
    train_df.to_csv("dataset/train.csv")
    val_df.to_csv("dataset/val.csv")
    test_df.to_csv("dataset/test.csv")
    schema_df.to_csv("dataset/docs.csv")