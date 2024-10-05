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
from rank_bm25 import BM25Okapi
import string



def arguments():
    parser = argparse.ArgumentParser(description='Model Arguments')
    parser.add_argument("-in","--input_file")
    parser.add_argument("-pk","--pk_file")
    parser.add_argument("-out","--output_file")
    parser.add_argument('-k', '--k',type=int)
    parser.add_argument('-cuda',"--cuda_device",type=int)
    

    args = parser.parse_args()

    return args

def get_embeddings(sentences,model):
    return model.encode(sentences)



number_of_samples = 500

def get_corpus(arrays):
    arrays = [i.lower() for i in arrays]
    arrays = [i.translate(str.maketrans('', '', string.punctuation)) for i in arrays]
    arrays = [i.split(" ") for i in arrays]
    return arrays


def BM25(corpus):
    bm25 = BM25Okapi(corpus)
    return bm25

def get_BM25_score(scorer,english,corpus):
    new_english = english.lower()
    new_english = new_english.translate(str.maketrans('', '', string.punctuation))
    new_english = new_english.split(" ")

    doc_scores = scorer.get_scores(new_english)

    return doc_scores

# def get_top_k_indices(numpy_array,k):
#     ind = np.argpartition(numpy_array, -k)[-k:]
#     return ind

if __name__ == "__main__":
    #Extract the arguments
    args = arguments()
    number_of_samples = args.k
    DEVICE = torch.device(f"cuda:{args.cuda_device}") if torch.cuda.is_available() else torch.device("cpu")

    #Open the datasets
    df = pd.read_csv(args.input_file)
    df.fillna(" ",inplace=True)
    doc_df = pd.read_csv(args.pk_file)

    question = df["nl"].tolist()
    gold_pk = df["pk_gold"].tolist()
    gold_pk = [i.split("<SEP>") for i in gold_pk]
    context_nl = doc_df["nl"].tolist()
    context = doc_df["pk"].tolist()
    

    context_ind_dict = {context[i]:i for i in range(0,len(context))}
    pk_till_now = {}
    pk_present_at_t = []
    bm25ranker = BM25(context)
    corpus = get_corpus(context)

    print("Embeddings extracted!")

    #Extract the top-k context
    extracted_context = []
    for i in tqdm(range(0,len(question))):
        # print(pk_till_now)
        temp_context = []
        temp_pk_present_at_t =[]
        score = get_BM25_score(bm25ranker,question[i],corpus)
        ind = np.argsort(score)
        ind = ind[::-1]

        #Select those actually present in the current partial knowledge
        cur_taken = 0
        temp_context = []

        for j in range(0,len(ind)):
            temp_pk = context[ind[j]]
            if temp_pk in pk_till_now:
                temp_context.append(temp_pk)
                cur_taken+=1
            if(cur_taken>=number_of_samples):
                break
        extracted_context.append("<SEP>".join(temp_context))

        #Add to the list for calculating knowledge recall
        for j in gold_pk[i]:
            if j in pk_till_now:
                temp_pk_present_at_t.append(j)
        
        pk_present_at_t.append("<SEP>".join(temp_pk_present_at_t))


        #Add to the partial knowledge all the gold context related to this row
        for j in gold_pk[i]:
            pk_till_now[j]=True
        

    
    df["parking"] = extracted_context
    df["knowledge_adjusted_gold"] = pk_present_at_t
    df.to_csv(args.output_file)

    