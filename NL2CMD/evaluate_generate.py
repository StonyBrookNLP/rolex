# Importing libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
import argparse
# Importing the T5 modules from huggingface/transformers
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from torch import cuda



parser = argparse.ArgumentParser(description='Model Arguments')
parser.add_argument("-file","--file")
parser.add_argument("-model","--model")
parser.add_argument("-pk","--is_lexicon",type=int)
parser.add_argument("-cuda","--cuda",type=int)
parser.add_argument("-sv","--save_file")

args = parser.parse_args()

filename = args.file
#filename = "percentage_ablation/"+args.file
if(args.is_lexicon==1):
  is_lexicon = True
else:
  is_lexicon = False
model_name = args.model

df = pd.read_csv(filename)

#print(df.columns)
print("Test Case:",filename)
print(model_name)

if(is_lexicon):
  df.fillna("",inplace=True)
  X = "NL2CMD\nNL:"+df["nl"]+"\nPK:"+df["parking"]
else:
  X = "NL2CMD\nNL:"+df["nl"]

X = X.to_list()
Y = df["fl"].to_list()

# Setting up the device for GPU usage
if(args.cuda==-1):
  device = 'cuda' if cuda.is_available() else 'cpu'
else:
  device = 'cuda:'+str(args.cuda) if cuda.is_available() else 'cpu'
print("Device:",device)

#Load the model and tokenizer from the folder
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.to(device)


#Encode and decode the target for best comparison in terms of BLEU
Y_encode = tokenizer.batch_encode_plus(
            Y,
            max_length=512,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
Y_encode = Y_encode["input_ids"]
print("Target prepped!")

#Encode the data
source = tokenizer.batch_encode_plus(
            X,
            max_length=256,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

source_id = source["input_ids"]
source_mask = source["attention_mask"]

#Do the predictions

model.eval()

predictions = []
targets = []
batch_size = 16

for i in range(0,len(X),batch_size):
  if(i+batch_size>=len(X)):
    temp_id = source_id[i:]
    temp_mask = source_mask[i:]
    y = Y_encode[i:]
  else:
    temp_id = source_id[i:i+batch_size]
    temp_mask = source_mask[i:i+batch_size]
    y = Y_encode[i:i+batch_size]
  
  temp_id = temp_id.to(device)
  temp_mask = temp_mask.to(device)
  
  generated_ids = model.generate(
    input_ids = temp_id,
    attention_mask = temp_mask,
    max_length=512,
    num_beams=5,
    repetition_penalty=2.5,
    length_penalty=1.0,
    early_stopping=True
  )

  print(i,"done!")

  prediction = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

  target = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in y]

  predictions = predictions + prediction
  targets = targets+target

if("parking" in filename or "ret" in filename):
  cols = {"Generated Text":predictions,"Actual Text":targets,"Adjustment":df["knowledge_adjusted_gold"],"Gold_PK":df["pk_gold"]}
elif("NLFL" in filename):
  cols = {"Generated Text":predictions,"Actual Text":targets,"Gold_PK":df["pk_gold"]}
else:
  cols = {"Generated Text":predictions,"Actual Text":targets}
new_df = pd.DataFrame(cols)
#new_df.to_csv("nfs_rfc_results/"+str(model_name))
new_df.to_csv(args.save_file)
print("Done!")