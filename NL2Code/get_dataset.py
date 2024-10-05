import numpy as np
import pandas as pd
import random
from datasets import load_dataset

# Load the dataset
conala_data = load_dataset("neulab/docprompting-conala","data")
conala_doc = load_dataset("neulab/docprompting-conala","docs")
train_df = pd.DataFrame(conala_data["train"])
val_df = pd.DataFrame(conala_data["validation"])
test_df = pd.DataFrame(conala_data["test"])
doc_df = pd.DataFrame(conala_doc["train"])


train_df.to_csv("dataset/train.csv")
test_df.to_csv("dataset/test.csv")
val_df.to_csv("dataset/val.csv")
doc_df.to_csv("dataset/docs.csv")