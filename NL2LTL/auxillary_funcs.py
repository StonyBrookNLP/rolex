import numpy as np
import random
from list_of_funcs import verbs_syn
from translation_rules import form_verb

def find(L,item):
    idx = []
    for i in range(0,len(L)):
        if(L[i]==item):
            idx.append(i)

    return idx

def get(L,item):
    for idx,it in enumerate(L):
        if(it==item):
            return idx

    return -1 

def possible_list_of_verb_phrases():
    x = list(verbs_syn.keys())
    y = {}
    for i in x:
        if(len(verbs_syn[i])>0):
            new_verb = random.choice(verbs_syn[i])
            new_verb = new_verb.replace("_"," ")
        else:
            new_verb = i
        y[i]=new_verb
        y[form_verb(i,"singular")]=form_verb(new_verb,"singular")
        y[form_verb(i,"past_tense")]=form_verb(new_verb,"past_tense")

    return y

def all_the_verbs():
    list_of_all_verbs = []

    for i in verbs_syn.keys():
        list_of_all_verbs.append(i)
        if(len(verbs_syn[i])>0):
            list_of_all_verbs += verbs_syn[i]
    
    # for i in list_of_all_verbs:
    #     list_of_all_verbs.append(form_verb(i,"singular"))
    #     list_of_all_verbs.append(form_verb(i,"past_tense"))
    
    return list_of_all_verbs


def get_modified_lexicon(lexicon):
    possible_phrases = possible_list_of_verb_phrases()

    temp = lexicon.copy()
    # temp_true = true_lexicon.copy()
    # print("FULL:",len(temp),temp)
    # print("PARTIAL:",len(temp_true),temp_true)

    for i in range(0,len(temp)):
        token = temp[i].split()
        # idx = get(temp_true,temp[i])
        # if(idx==-1):
        #     token_true = []
        # else:
        #     token_true = temp_true[idx].split()
        

        for j in range(0,len(token)):
            # true_idx_if_exist = get(token_true,j)
            if(token[j] in possible_phrases.keys()):
                token[j] = possible_phrases[token[j]]
                # if(true_idx_if_exist!=-1):
                #     token_true[true_idx_if_exist] = possible_phrases[token[j]]
        token = " ".join(token)
        # token_true = " ".join(token_true)
        temp[i] = token
        # temp_true = token_true
    #return temp,temp_true
    return temp