import pandas as pd
import numpy as np
import argparse
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import AdamW, get_scheduler
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses,util
from tqdm import tqdm

def arguments():
    parser = argparse.ArgumentParser(description='Model Arguments')
    parser.add_argument("-model","--model")
    parser.add_argument("-in","--input_file")
    parser.add_argument("-pk","--pk_file")
    parser.add_argument("-out","--output_file")
    parser.add_argument('-k', '--k',type=int)
    parser.add_argument('-cuda',"--cuda_device",type=int)

    args = parser.parse_args()

    return args

def get_embeddings(sentences,model):
    return model.encode(sentences)

if __name__ == "__main__":
    #Extract the arguments
    args = arguments()
    number_of_samples = args.k
    DEVICE = torch.device(f"cuda:{args.cuda_device}") if torch.cuda.is_available() else torch.device("cpu")

    #Open the datasets
    df = pd.read_csv(args.input_file)
    doc_df = pd.read_csv(args.pk_file)

    question = df["nl"].tolist()
    context_nl = doc_df["nl"].tolist()
    context = doc_df["pk"].tolist()
    

    if("test" in args.input_file):
        doc_df_rnf = pd.read_csv("dataset/test_rnf_pk.csv")
        context_rnf = doc_df_rnf["pk"].tolist()
        rnf = df["rnf"].tolist()

    #Initialize the models 
    model = SentenceTransformer(args.model,device=DEVICE)


    #Get all the context embeddings 
    question_embeddings = get_embeddings(question,model)
    context_embeddings = get_embeddings(context_nl,model)

    print("Embeddings extracted!")

    #Extract the top-k context
    extracted_context = []
    for i in tqdm(range(0,len(question))):
        temp_context = []
        cos_sim = util.cos_sim(context_embeddings,question_embeddings[i]).squeeze().numpy()
        ind = np.argsort(cos_sim)
        ind = ind[-number_of_samples:]
        if("test" in args.input_file):
            if(rnf[i]==True):
                print(i)
                temp_context = [context_rnf[j] for j in ind]
            else:
                temp_context = [context[j] for j in ind]
        else:
            temp_context = [context[j] for j in ind]
        temp_context = temp_context[::-1]
        extracted_context.append("<SEP>".join(temp_context))
    
    df["pk_retrieved"] = extracted_context
    df.to_csv(args.output_file)

    