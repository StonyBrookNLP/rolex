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
import ast


class FunctionCallExtractor(ast.NodeVisitor):
    def __init__(self):
        self.called_functions = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.called_functions.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.called_functions.append(node.func.attr)
        self.generic_visit(node)

def extract_function_calls(code):
    tree = ast.parse(code)
    extractor = FunctionCallExtractor()
    extractor.visit(tree)
    return extractor.called_functions

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

    #Initialize the models 
    model = SentenceTransformer(args.model,device=DEVICE)

    #Get all the context embeddings 
    question_embeddings = get_embeddings(question,model)
    context_embeddings = get_embeddings(context_nl,model)

    print("Embeddings extracted!")

    #Extract the top-k context
    extracted_context = []
    for i in tqdm(range(0,len(question))):
        # print(pk_till_now)
        temp_context = []
        temp_pk_present_at_t =[]
        cos_sim = util.cos_sim(context_embeddings,question_embeddings[i]).squeeze().numpy()
        ind = np.argsort(cos_sim)
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

    