import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from transformers import AdamW
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
import argparse


def arguments():
    parser = argparse.ArgumentParser(description='Model Arguments')
    parser.add_argument("-input_dataset","--dataset")
    parser.add_argument("-model_to_train","--model_name")
    parser.add_argument("-save_model_path","--save_path")
    parser.add_argument('-cuda','--cuda_device')
    parser.add_argument('-bs','--batch_size',type=int)
    parser.add_argument('-lr','--learning_rate',type=float)
    parser.add_argument('-ep','--epochs',type=int)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    #Getting the hyperparameters
    args = arguments()

    # Experiment hyperparameters
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    INPUT_FILE = args.dataset
    SAVE_PATH = args.save_path
    DEVICE = torch.device(f"cuda:{args.cuda_device}") if torch.cuda.is_available() else torch.device("cpu")

    #Load the dataframe and create the training examples
    df = pd.read_csv(INPUT_FILE)
    df.fillna("",inplace=True)

    question_list = df["sentence_1"]
    context_list = df["sentence_2"]

    train_examples = []

    for i in range(0,len(question_list)):
        input_example = InputExample( texts = [question_list[i],context_list[i]] )
        train_examples.append(input_example)

    #Initialize the model
    model = SentenceTransformer(args.model_name,device=DEVICE)


    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=EPOCHS,show_progress_bar=True)

    model.save(SAVE_PATH)
