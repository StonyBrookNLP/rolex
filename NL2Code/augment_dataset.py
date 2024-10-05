import numpy as np
import pandas as pd
import random
import ast
import re

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


def augment_dataset(df,doc_dict):
    #Get the relevant document ids
    relevant_doc_ids = df["oracle_man"].tolist()
    python_code = df["cmd"].tolist()
    relevant_doc_ids = [item[1:-1] for item in relevant_doc_ids]
    relevant_doc_ids = [item.split(",") for item in relevant_doc_ids]

    for i in range(0,len(relevant_doc_ids)):
        for j in range(0,len(relevant_doc_ids[i])):
            relevant_doc_ids[i][j] = relevant_doc_ids[i][j].strip()[1:-1]

    pk_base = {}
    pk_gold = []
    open_vocabs = []
    unique_open_vocabs = {}

    #Augment with the relevant partial knowledge and extract the openvocabs
    for i in range(0,len(relevant_doc_ids)):
        temp_pk_gold = []
        func_names = extract_function_calls(python_code[i])
        for ovc in func_names:
            unique_open_vocabs[ovc]=True
        temp_open_vocab = []
        for j in range(0,len(relevant_doc_ids[i])):
            temp_fl = relevant_doc_ids[i][j]
            
            for k in func_names:
                if k in temp_fl:
                    temp_open_vocab.append(k)

            if(temp_fl==''):
                continue
            temp_nl = doc_dict[temp_fl]
            temp_pk = f"{temp_nl}=>{temp_fl}"
            pk_base[temp_pk] = True
            temp_pk_gold.append(temp_pk)
        temp_open_vocab = list(set(temp_open_vocab))
        open_vocabs.append(temp_open_vocab)
        pk_gold.append(temp_pk_gold)
    
    open_vocabs = ["<SEP>".join(i) for i in open_vocabs]
    pk_gold = ["<SEP>".join(i) for i in pk_gold]

    # print(len(unique_open_vocabs))
    # print(len(pk_base))
    # print()

    df.rename(columns={"nl":"nl","cmd":"fl"},inplace=True)
    df["pk_gold"] = pk_gold
    df["OVC"] = open_vocabs

    pk_base = [i for i in pk_base.keys()]

    return df,pk_base

def save_pk(pk_list,pk_filename):
    nl = []
    fl = []
    for i in pk_list:
        t1,t2 = i.split("=>")
        nl.append(t1)
        fl.append(t2)
    
    df_pk = pd.DataFrame({"nl":nl,"fl":fl,"pk":pk_list})
    df_pk.to_csv(pk_filename)



if __name__ == "__main__":
    #Process the documents
    docs_df = pd.read_csv("dataset/docs.csv")
    train_df = pd.read_csv("dataset/train.csv")
    val_df = pd.read_csv("dataset/val.csv")
    test_df = pd.read_csv("dataset/test.csv")
    length_of_doc = 200


    #Get the dictionary of documents
    ids = docs_df["doc_id"].tolist()
    contents = docs_df["doc_content"].tolist()
    contents = [re.sub(' +', ' ',i) for i in contents]
    doc_dict = {ids[i]:contents[i][:length_of_doc].strip() for i in range(0,len(ids))}
    
    #Augment the dataset and create the partial knowledge base for each dataset
    train_df, train_pk = augment_dataset(train_df,doc_dict)
    val_df, val_pk = augment_dataset(val_df,doc_dict)
    test_df, test_pk = augment_dataset(test_df,doc_dict)


    train_df.to_csv("dataset/train_augmented.csv")
    val_df.to_csv("dataset/val_augmented.csv")
    test_df.to_csv("dataset/test_augmented.csv")
    save_pk(train_pk,"dataset/train_pk.csv")
    save_pk(val_pk,"dataset/val_pk.csv")
    save_pk(test_pk,"dataset/test_pk.csv")

    print("Partial Knowledge Augmentation Done!")

