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
import faiss



def arguments():
    parser = argparse.ArgumentParser(description='Model Arguments')
    parser.add_argument("-in","--input_file")
    parser.add_argument("-pk","--pk_file")
    parser.add_argument("-in_emb","--input_embed")
    parser.add_argument("-pk_emb","--pk_embed")
    parser.add_argument("-out","--output_file")
    parser.add_argument('-k', '--k',type=int)
    parser.add_argument('-cuda',"--cuda_device",type=int)
    

    args = parser.parse_args()

    return args

def get_faiss_index(context_embeddings):
    x = context_embeddings.copy()
    index = faiss.IndexFlatL2(x.shape[1])
    faiss.normalize_L2(x)
    index.add(x)
    len_of_index = x.shape[0]
    return index,len_of_index

def get_faiss_search(input_embedding,faiss_index,len_of_index):
    distances,indices = faiss_index.search(input_embedding,len_of_index)
    return indices



if __name__ == "__main__":
    #Extract the arguments
    args = arguments()
    number_of_samples = args.k

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

    # #Initialize the models 
    # model = SentenceTransformer(args.model,device=DEVICE)

    #Get all the context embeddings 
    question_embeddings = np.load(args.input_embed)
    question_embeddings = question_embeddings.astype("float32")
    context_embeddings = np.load(args.pk_embed)
    context_embeddings = context_embeddings.astype("float32")
    faiss_index,index_len = get_faiss_index(context_embeddings)
    # print("Index Length:",index_len)

    print("Embeddings extracted!")

    #Extract the top-k context
    extracted_context = []
    for i in tqdm(range(0,len(question))):
        # print(pk_till_now)
        temp_context = []
        temp_pk_present_at_t =[]
        # cos_sim = util.cos_sim(context_embeddings,question_embeddings[i]).squeeze().numpy()
        ind = get_faiss_search(np.expand_dims(question_embeddings[i], axis=0),faiss_index,index_len)[0]
        # ind = ind[::-1]

        #Select those actually present in the current partial knowledge
        cur_taken = 0
        temp_context = []

        for j in range(0,len(ind)):
            # print(j)
            # print(ind)
            # print(len(ind))
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

    