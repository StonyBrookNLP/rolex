import random
import numpy as np

from rules import stl_rules
from translation_rules import generate_translate_scheme,sample_translation,paraphrase_if,generate_paraphrase_scheme
from lexicon import create_lexicon,sample_lexicon
from auxillary_funcs import get_modified_lexicon
import pandas as pd

number_of_datapoints = 2000 # Number of datapoints to generate
template_limit = 60 # Determines how long a formula should be
construct_space = 2 # The space of the verbs and nouns to use. Construct 1 is test distribution. Construct 2 is training distribution.
creating_train = False # Flag to determine whether we are generating training or testing datasets.

print(number_of_datapoints,template_limit,construct_space,creating_train)

R = stl_rules(construct_space)


def save_baseline(list_of_template,list_of_stl,list_of_translation):
    df=pd.DataFrame({"template":list_of_template,"fl":list_of_stl,"nl":list_of_translation})
    df.to_csv("dataset/baseline_train.csv")

def save_RAG(list_of_template,list_of_modified_stl,list_of_translation,list_of_lexicon):
    df = pd.DataFrame({"template":list_of_template,"fl":list_of_modified_stl,"nl":list_of_translation,"pk_gold":list_of_lexicon})
    df.to_csv("dataset/RAG_train.csv")


def save_baseline_test(list_of_template,list_of_stl,list_of_translation,list_of_paraphrases,list_of_modified_stl,list_of_rnf_stl,list_of_rnf_lexicon):

    df=pd.DataFrame({"template":list_of_template,"fl":list_of_stl,"nl":list_of_translation})
    if(construct_space==1):
        df.to_csv("dataset/baseline_test_simple_indist.csv")
    elif(construct_space==2):
        df.to_csv("dataset/baseline_test_simple_outdist.csv")
    else:
        df.to_csv("dataset/baseline_test_simple_full.csv")
    
    df=pd.DataFrame({"template":list_of_template,"fl":list_of_stl,"nl":list_of_paraphrases})
    if(construct_space==1):
        df.to_csv("dataset/baseline_test_paraphrase_indist.csv")
    elif(construct_space==2):
        df.to_csv("dataset/baseline_test_paraphrase_outdist.csv")
    else:
        df.to_csv("dataset/baseline_test_paraphrase_full.csv")

    df=pd.DataFrame({"template":list_of_template,"fl":list_of_rnf_stl,"nl":list_of_translation})
    if(construct_space==1):
        df.to_csv("dataset/baseline_test_rnf_indist.csv")
    elif(construct_space==2):
        df.to_csv("dataset/baseline_test_rnf_outdist.csv")
    else:
        df.to_csv("dataset/baseline_test_rnf_full.csv")

    
    df=pd.DataFrame({"template":list_of_template,"fl":list_of_modified_stl,"nl":list_of_paraphrases})
    if(construct_space==1):
        df.to_csv("dataset/baseline_test_all_indist.csv")
    elif(construct_space==2):
        df.to_csv("dataset/baseline_test_all_outdist.csv")
    else:
        df.to_csv("dataset/baseline_test_all_full.csv")

def save_RAG_test(list_of_template,list_of_modified_stl,list_of_translation,list_of_lexicon,list_of_paraphrases,list_of_modified_lexicon,list_of_rnf_stl,list_of_rnf_lexicon):

    df = pd.DataFrame({"template":list_of_template,"fl":list_of_modified_stl,"nl":list_of_translation,"pk_gold":list_of_lexicon})
    if(construct_space==1):
        df.to_csv("dataset/RAG_test_simple_indist.csv")
    elif(construct_space==2):
        df.to_csv("dataset/RAG_test_simple_outdist.csv")
    else:
        df.to_csv("dataset/RAG_test_simple_full.csv")
    

    df = pd.DataFrame({"template":list_of_template,"fl":list_of_modified_stl,"nl":list_of_paraphrases,"pk_gold":list_of_lexicon})
    if(construct_space==1):
        df.to_csv("dataset/RAG_test_paraphrase_indist.csv")
    elif(construct_space==2):
        df.to_csv("dataset/RAG_test_paraphrase_outdist.csv")
    else:
        df.to_csv("dataset/RAG_test_paraphrase_full.csv")

    
    df = pd.DataFrame({"template":list_of_template,"fl":list_of_rnf_stl,"nl":list_of_translation,"pk_gold":list_of_rnf_lexicon})
    if(construct_space==1):
        df.to_csv("dataset/RAG_test_rnf_indist.csv")
    elif(construct_space==2):
        df.to_csv("dataset/RAG_test_rnf_outdist.csv")
    else:
        df.to_csv("dataset/RAG_test_rnf_full.csv")

    df = pd.DataFrame({"template":list_of_template,"fl":list_of_modified_stl,"nl":list_of_paraphrases,"pk_gold":list_of_lexicon})
    if(construct_space==1):
        df.to_csv("dataset/RAG_test_all_indist.csv")
    elif(construct_space==2):
        df.to_csv("dataset/RAG_test_all_outdist.csv")
    else:
        df.to_csv("dataset/RAG_test_all_full.csv")



def generate_an_STL():
    parse_template_queue = []
    parse_stl_queue = []

    generated_otherwise = False
    generated_if = False

    template = []
    stl_formula = []
    translation = []

    parse_template_queue.append("start")
    parse_stl_queue.append("start")

    while(parse_template_queue):
        front_template = parse_template_queue.pop(0)
        front_stl = parse_stl_queue.pop(0)

        if(front_template in R.list_of_terminals):
            template.append(front_template)
            stl_formula.append(front_stl)
        else:
            next_template_parse,next_stl_parse,rule_id = R.generate_next_parse(front_template)

            if(rule_id==1):
                generated_if = True

            if(rule_id==2):
                generated_otherwise = True

            parse_template_queue = next_template_parse+parse_template_queue
            parse_stl_queue = next_stl_parse+parse_stl_queue

    translation_scheme = generate_translate_scheme(template,stl_formula,creating_train)
    paraphrase_scheme = generate_paraphrase_scheme(template,stl_formula)


    if(generated_otherwise):
        template,stl_formula = R.handle_otherwise(template,stl_formula)

    if(generated_if):
        sample = random.sample([1,2,3,4],1)
        if(sample==1):
            translation_scheme = paraphrase_if(translation_scheme)
        if(sample==4):
            paraphrase_scheme = paraphrase_if(paraphrase_scheme)

    translation = sample_translation(translation_scheme,template)
    paraphrase = sample_translation(paraphrase_scheme,template)

    custom_lexicon,modified_stl = sample_lexicon(template,stl_formula,creating_train)
    rnf_lexicon,rnf_stl = sample_lexicon(template,stl_formula,creating_train,True)
    modified_lexicon = get_modified_lexicon(custom_lexicon)

    template_length = len(template)
    return " ".join(template)," ".join(stl_formula),translation," ".join(modified_stl),"<SEP>".join(custom_lexicon),paraphrase,"<SEP>".join(modified_lexicon),template_length," ".join(rnf_stl),"<SEP>".join(rnf_lexicon)

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    list_of_template = []
    list_of_stl = []
    list_of_translation = []
    list_of_lexicon = []
    list_of_modified_stl = []
    list_of_modified_lexicon = []
    list_of_true_lexicon = []
    list_of_modified_true_lexicon = []
    list_of_paraphrases = []
    list_of_rnf_stl = []
    list_of_rnf_lexicon = []

    i = 0
    average_template_size = 0
    max_template_size = -1
    min_template_size = 1000000
    above_100 = 0
    while(i<number_of_datapoints):
        template,stl,translation,modified_stl,custom_lexicon,paraphrase,modified_lexicon,template_length,rnf_stl,rnf_lexicon = generate_an_STL()
        # average_template_size += template_length
        # max_template_size = max(max_template_size,template_length)
        # min_template_size = min(min_template_size,template_length)
        if(template_length>template_limit):
            continue

        list_of_template.append(template)
        list_of_stl.append(stl)
        list_of_translation.append(translation)
        list_of_modified_stl.append(modified_stl)
        list_of_lexicon.append(custom_lexicon)
        list_of_paraphrases.append(paraphrase)
        list_of_modified_lexicon.append(modified_lexicon)
        list_of_rnf_stl.append(rnf_stl)
        list_of_rnf_lexicon.append(rnf_lexicon)

        i+=1

        if(i%1000==0):
            print(i,"data points done!")

    if(creating_train):
        save_baseline(list_of_template,list_of_stl,list_of_translation)
        save_RAG(list_of_template,list_of_modified_stl,list_of_translation,list_of_lexicon)
    else:
        save_baseline_test(list_of_template,list_of_stl,list_of_translation,list_of_paraphrases,list_of_modified_stl,list_of_rnf_stl,list_of_rnf_lexicon)
        save_RAG_test(list_of_template,list_of_modified_stl,list_of_translation,list_of_lexicon,list_of_paraphrases,list_of_modified_lexicon,list_of_rnf_stl,list_of_rnf_lexicon)