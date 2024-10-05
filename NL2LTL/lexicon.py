import numpy as np
import pandas as pd
import random
from list_of_funcs import verbs,verbs_syn,varib_noun_dict
from translation_rules import form_verb
from name_generation import generate_variable
from auxillary_funcs import all_the_verbs

per_sample_lexicons = 8
types_of_verbs = ["#VERB0","#VERB1","#VERB2","#VERB3","#VERB4"]

def check_elements(A, B):
    for element in A:
        if element in B:
            return True
    return False

def create_lexicon(type,verb,train,is_rnf=False):
    # func_id = random.randint(0,10000)
    #func_id = "func"+str(func_id)
    # func_id = verb
    #func_id = generate_variable()
    func_id = verb
    if(not train and is_rnf):
        idx = 5*verbs.index(verb)
        if(type == "#VERB0"):
            func_id = f"func{idx}"
        if(type == "#VERB1"):
            func_id = f"func{idx+1}"
        if(type == "#VERB2"):
            func_id = f"func{idx+2}"
        if(type == "#VERB3"):
            func_id = f"func{idx+3}"
        if(type == "#VERB4"):
            func_id = f"func{idx+4}"

    if(not train):

        if(type == "#VERB0"): 
            translation_rule = "agent "+form_verb(verb,"singular")+" => "
            grounding_phrase = translation_rule
            translation_rule += str(func_id) +" ( E ) "

        if(type == "#VERB1"):
            translation_rule= "A is "+form_verb(verb,"past_tense")+" => "
            grounding_phrase = translation_rule
            translation_rule += str(func_id) + " ( E ) WITH E . A"

        if (type == "#VERB2"):
            translation_rule = "A is " + form_verb(verb, "past_tense")+" with B"+" => "
            grounding_phrase = translation_rule
            translation_rule += str(func_id) + " ( E ) WITH E . A WITH E . B"

        if (type == "#VERB3"):
            translation_rule = "A is " + form_verb(verb, "past_tense")+" with B and C"+" => "
            grounding_phrase = translation_rule
            translation_rule += str(func_id) + " ( E ) WITH E . A WITH E . B WITH E . C"

        if (type == "#VERB4"):
            translation_rule = "A is " + form_verb(verb, "past_tense")+" with B and C and D"+" => "
            grounding_phrase = translation_rule
            translation_rule += str(func_id) + " ( E ) WITH E . A WITH E . B WITH E . C WITH E . D"
    else:
        translation_phrases = []
        if(type == "#VERB0"):
            translation_phrases.append("agent "+form_verb(verb,"singular")+" => ")
            # translation_phrases.append(form_verb(verb,"singular")+" => ")

            # list_of_synonyms = verbs_syn[verb]

            # for synonym in list_of_synonyms:
            #     translation_phrases.append("agent "+form_verb(synonym,"singular")+" => ")
            #     translation_phrases.append(form_verb(synonym,"singular")+" => ")
            
            translation_rule = random.choice(translation_phrases)
            grounding_phrase = translation_rule
            translation_rule += str(func_id) +" ( E ) "
        
        if(type == "#VERB1"):
            translation_phrases.append("A is " + form_verb(verb, "past_tense")+" => ")
            # translation_phrases.append(verb + " A"+" => ")

            # list_of_synonyms = verbs_syn[verb]

            # for synonym in list_of_synonyms:
            #     translation_phrases.append("A is " + form_verb(synonym, "past_tense")+" => ")
            #     translation_phrases.append(synonym + " A"+" => ")
            
            translation_rule = random.choice(translation_phrases)
            grounding_phrase = translation_rule
            translation_rule += str(func_id) + " ( E ) WITH E . A"

        if (type == "#VERB2"):
            translation_phrases.append("A is " + form_verb(verb, "past_tense")+" with B"+" => ")
            # translation_phrases.append(verb + " A with B =>")

            # list_of_synonyms = verbs_syn[verb]

            # for synonym in list_of_synonyms:
            #     translation_phrases.append("A is " + form_verb(synonym, "past_tense")+" with B"+" => ")
            #     translation_phrases.append(synonym + " A with B"+" => ")
            
            translation_rule = random.choice(translation_phrases)
            grounding_phrase = translation_rule
            translation_rule += str(func_id) + " ( E ) WITH E . A WITH E . B"

        if (type == "#VERB3"):
            translation_phrases.append("A is " + form_verb(verb, "past_tense")+" with B and C"+" => ")
            # translation_phrases.append(verb + " A with B and C =>")
            # translation_phrases.append(verb + " A and B with C =>")

            # list_of_synonyms = verbs_syn[verb]

            # for synonym in list_of_synonyms:
            #     translation_phrases.append("A is " + form_verb(synonym, "past_tense")+" with B and C"+" => ")
            #     translation_phrases.append(synonym + " A with B and C =>")
            #     translation_phrases.append(synonym + " A and B with C =>")
            
            translation_rule = random.choice(translation_phrases)
            grounding_phrase = translation_rule
            translation_rule += str(func_id) + " ( E ) WITH E . A WITH E . B WITH E . C"

        if (type == "#VERB4"):
            translation_phrases.append("A is " + form_verb(verb, "past_tense")+" with B and C and D"+" => ")
            # translation_phrases.append(verb + " A with B and C and D =>")
            # translation_phrases.append(verb + " A and B with C and D =>")
            # translation_phrases.append(verb + " A and B and C with D =>")

            # list_of_synonyms = verbs_syn[verb]

            # for synonym in list_of_synonyms:
            #     translation_phrases.append("A is " + form_verb(synonym, "past_tense")+" with B and C and D"+" => ")
            #     translation_phrases.append(synonym + " A with B and C and D =>")
            #     translation_phrases.append(synonym + " A and B with C and D =>")
            #     translation_phrases.append(synonym + " A and B and C with D =>")
            
            translation_rule = random.choice(translation_phrases)
            grounding_phrase = translation_rule
            translation_rule += str(func_id) + " ( E ) WITH E . A WITH E . B WITH E . C WITH E . D"


    return translation_rule, func_id,grounding_phrase


def sample_lexicon(orig_template,orig_stl,train,is_rnf=False):
    modified_stl = orig_stl.copy()
    template = orig_template.copy()
    custom_lexicon = []
    number_of_rules = 0
    verbs_taken = []
    dict_of_func_id = {}

    rnf_flag = False

    if(is_rnf):
        rnf_flag = True

    for i in range(0,len(template)-1):
        if("#VERB" in template[i]):
            prob = random.choice([1,2,3])
            if(prob >= 1):
                verb = orig_stl[i]
                if (verb in dict_of_func_id.keys()):
                    modified_stl[i] = dict_of_func_id[verb]
                    continue
                verbs_taken.append(verb)
                type_of_verb = template[i]
                lexicon_rule,function_id,grounding_phrase = create_lexicon(type_of_verb,verb,train,rnf_flag)
                custom_lexicon.append(lexicon_rule)
                modified_stl[i] = function_id
                number_of_rules+=1

                dict_of_func_id[verb] = function_id
                

                # lexicon_ambiguity_prob = random.choice([i for i in range(0,10)])

                # if(lexicon_ambiguity_prob>=0):
                #     verb = orig_stl[i]
                #     type_of_verb = template[i]
                #     while(type_of_verb==template[i]):
                #         type_of_verb = random.choice(types_of_verbs)
                #     fake_rule = create_fake_lexicon(type_of_verb,grounding_phrase,function_id)
                #     custom_lexicon.append(fake_rule)
                #     number_of_rules+=10
                    
        
        if("#GET"==template[i]):
            get_varib = orig_stl[i]
            idx = get_varib.find("(")
            get,varib = get_varib[:idx],get_varib[idx+1:-1]
            get = get[4:]

            if(get in varib_noun_dict.keys()):
                lexicon_rule = f"{varib_noun_dict[get]} => {get}"
                custom_lexicon.append(lexicon_rule)
            
            if(varib in varib_noun_dict.keys()):
                lexicon_rule = f"{varib_noun_dict[varib]} => {varib}"
                custom_lexicon.append(lexicon_rule)

            # prob = random.choice([1,2,3,4])
            # if(prob==1):
            #     get_varib = orig_stl[i]
            #     idx = get_varib.find("(")
            #     get,variable = get_varib[:idx],get_varib[idx:]
            #     get_varib = get[4:]
            #     if(get_varib in dict_of_func_id.keys()):
            #         modified_stl[i] = dict_of_func_id[get_varib]
            #         continue
            #     new_get = "get_"+generate_variable()
            #     rules = [f'the {get_varib} of X',f'X {get_varib}',f'{get_varib} property of X']
            #     lexicon_rule = random.choice(rules)+" => "+new_get+"(X)"
            #     custom_lexicon.append(lexicon_rule)
            #     modified_stl[i] = new_get+variable
            #     number_of_rules+=1

            #     dict_of_func_id[get_varib] = new_get+variable

        if("#VARIABLE" == template[i]):
            if(orig_stl[i] in varib_noun_dict):
                variable = orig_stl[i]
                lexicon_rule = f"{varib_noun_dict[variable]} => {variable}"
                # if(variable in dict_of_func_id.keys()):
                #     modified_stl[i] = dict_of_func_id[variable]
                #     continue
                # new_variable = generate_variable()
                # lexicon_rule = variable+" => "+new_variable
                custom_lexicon.append(lexicon_rule)
                # modified_stl[i] = new_variable
                number_of_rules+=1

                # dict_of_func_id[variable] = new_variable

        if ("#VARIABLE_BOUND" == template[i] and ">=" == template[i+1]):
            if(orig_stl[i] in varib_noun_dict):
                variable = orig_stl[i]
                lexicon_rule = f"{varib_noun_dict[variable]} => {variable}"

                # if(variable in dict_of_func_id.keys()):
                #     modified_stl[i] = dict_of_func_id[variable]
                #     modified_stl[i+4] = dict_of_func_id[variable]
                #     template[i+4]="#DONE"
                #     continue

                # new_variable = generate_variable()
                # lexicon_rule = variable + " => " + new_variable
                custom_lexicon.append(lexicon_rule)
                modified_stl[i] = variable

                modified_stl[i+4] = variable
                template[i+4]="#DONE"
                number_of_rules+=1

                # dict_of_func_id[variable] = new_variable

    
    # #Include the synonyms as well
    # new_verbs_taken = []
    # for i in verbs_taken:
    #     new_verbs_taken.append(i)
    #     new_verbs_taken = new_verbs_taken+verbs_syn[i]
    
    
    # verbs_taken = new_verbs_taken

    # true_lexicon = custom_lexicon.copy()

    # if(number_of_rules>0 and number_of_rules<per_sample_lexicons):
    #     i = number_of_rules
    #     while(i<per_sample_lexicons):
    #         type_of_verb = random.sample(types_of_verbs,1)[0]
    #         verb = random.sample(verbs,1)[0]
    #         verb_syn_temp = verbs_syn[verb]
    #         if(verb in verbs_taken):
    #             continue
    #         # if(check_elements(verb_syn_temp,verbs_taken)):
    #         #     continue
    #         lexicon_rule,D,G = create_lexicon(type_of_verb,verb,train)
    #         custom_lexicon.append(lexicon_rule)
    #         i+=1
    # #print(len(verbs_taken),len(verbs_taken)-len(set(verbs_taken)))
    # # Z = " ".join(custom_lexicon).split()
    # # print(Z,len(Z)-len(set(Z)))
    random.shuffle(custom_lexicon)
    return custom_lexicon,modified_stl






def create_lexicon_without_indirect(type,verb,train):
    func_id = random.randint(0,10000)
    func_id = "func"+str(func_id)
    #func_id = generate_variable()
    if(not train):

        if(type == "#VERB0"): 
            translation_rule = "agent "+form_verb(verb,"singular")+" => "
            grounding_phrase = translation_rule
            translation_rule += str(func_id) +" ( E ) "

        if(type == "#VERB1"):
            translation_rule= "A is "+form_verb(verb,"past_tense")+" => "
            grounding_phrase = translation_rule
            translation_rule += str(func_id) + " ( E ) WITH E . A"

        if (type == "#VERB2"):
            translation_rule = "A is " + form_verb(verb, "past_tense")+" with B"+" => "
            grounding_phrase = translation_rule
            translation_rule += str(func_id) + " ( E ) WITH E . A WITH E . B"

        if (type == "#VERB3"):
            translation_rule = "A is " + form_verb(verb, "past_tense")+" with B and C"+" => "
            grounding_phrase = translation_rule
            translation_rule += str(func_id) + " ( E ) WITH E . A WITH E . B WITH E . C"

        if (type == "#VERB4"):
            translation_rule = "A is " + form_verb(verb, "past_tense")+" with B and C and D"+" => "
            grounding_phrase = translation_rule
            translation_rule += str(func_id) + " ( E ) WITH E . A WITH E . B WITH E . C WITH E . D"
    else:
        translation_phrases = []
        if(type == "#VERB0"):
            translation_phrases.append("agent "+form_verb(verb,"singular")+" => ")
            # translation_phrases.append(form_verb(verb,"singular")+" => ")

            # list_of_synonyms = verbs_syn[verb]

            # for synonym in list_of_synonyms:
            #     translation_phrases.append("agent "+form_verb(synonym,"singular")+" => ")
            #     translation_phrases.append(form_verb(synonym,"singular")+" => ")
            
            translation_rule = random.choice(translation_phrases)
            grounding_phrase = translation_rule
            translation_rule += str(func_id) +" ( E ) "
        
        if(type == "#VERB1"):
            translation_phrases.append("A is " + form_verb(verb, "past_tense")+" => ")
            # translation_phrases.append(verb + " A"+" => ")

            # list_of_synonyms = verbs_syn[verb]

            # for synonym in list_of_synonyms:
            #     translation_phrases.append("A is " + form_verb(synonym, "past_tense")+" => ")
            #     translation_phrases.append(synonym + " A"+" => ")
            
            translation_rule = random.choice(translation_phrases)
            grounding_phrase = translation_rule
            translation_rule += str(func_id) + " ( E ) WITH E . A"

        if (type == "#VERB2"):
            translation_phrases.append("A is " + form_verb(verb, "past_tense")+" with B"+" => ")
            # translation_phrases.append(verb + " A with B =>")

            # list_of_synonyms = verbs_syn[verb]

            # for synonym in list_of_synonyms:
            #     translation_phrases.append("A is " + form_verb(synonym, "past_tense")+" with B"+" => ")
            #     translation_phrases.append(synonym + " A with B"+" => ")
            
            translation_rule = random.choice(translation_phrases)
            grounding_phrase = translation_rule
            translation_rule += str(func_id) + " ( E ) WITH E . A WITH E . B"

        if (type == "#VERB3"):
            translation_phrases.append("A is " + form_verb(verb, "past_tense")+" with B and C"+" => ")
            # translation_phrases.append(verb + " A with B and C =>")
            # translation_phrases.append(verb + " A and B with C =>")

            # list_of_synonyms = verbs_syn[verb]

            # for synonym in list_of_synonyms:
            #     translation_phrases.append("A is " + form_verb(synonym, "past_tense")+" with B and C"+" => ")
            #     translation_phrases.append(synonym + " A with B and C =>")
            #     translation_phrases.append(synonym + " A and B with C =>")
            
            translation_rule = random.choice(translation_phrases)
            grounding_phrase = translation_rule
            translation_rule += str(func_id) + " ( E ) WITH E . A WITH E . B WITH E . C"

        if (type == "#VERB4"):
            translation_phrases.append("A is " + form_verb(verb, "past_tense")+" with B and C and D"+" => ")
            # translation_phrases.append(verb + " A with B and C and D =>")
            # translation_phrases.append(verb + " A and B with C and D =>")
            # translation_phrases.append(verb + " A and B and C with D =>")

            # list_of_synonyms = verbs_syn[verb]

            # for synonym in list_of_synonyms:
            #     translation_phrases.append("A is " + form_verb(synonym, "past_tense")+" with B and C and D"+" => ")
            #     translation_phrases.append(synonym + " A with B and C and D =>")
            #     translation_phrases.append(synonym + " A and B with C and D =>")
            #     translation_phrases.append(synonym + " A and B and C with D =>")
            
            translation_rule = random.choice(translation_phrases)
            grounding_phrase = translation_rule
            translation_rule += str(func_id) + " ( E ) WITH E . A WITH E . B WITH E . C WITH E . D"


    return translation_rule, str(func_id),grounding_phrase


def sample_lexicon_without_indirect(orig_template,orig_stl,train):
    modified_stl = orig_stl.copy()
    template = orig_template.copy()
    custom_lexicon = []
    number_of_rules = 0
    verbs_taken = []
    dict_of_func_id = {}
    for i in range(0,len(template)-1):
        if("#VERB" in template[i]):
            prob = random.choice([1,2,3])
            if(prob >= 1):
                verb = orig_stl[i]
                if (verb in dict_of_func_id.keys()):
                    modified_stl[i] = dict_of_func_id[verb]
                    continue
                verbs_taken.append(verb)
                type_of_verb = template[i]
                lexicon_rule,function_id,grounding_phrase = create_lexicon(type_of_verb,verb,train)
                custom_lexicon.append(lexicon_rule)
                modified_stl[i] = function_id
                number_of_rules+=1

                dict_of_func_id[verb] = function_id

                # lexicon_ambiguity_prob = random.choice([i for i in range(0,10)])

                # if(lexicon_ambiguity_prob>=0):
                #     verb = orig_stl[i]
                #     type_of_verb = template[i]
                #     while(type_of_verb==template[i]):
                #         type_of_verb = random.choice(types_of_verbs)
                #     fake_rule = create_fake_lexicon(type_of_verb,grounding_phrase,function_id)
                #     custom_lexicon.append(fake_rule)
                #     number_of_rules+=10
                    
        
        if("#GET"==template[i]):
            prob = random.choice([1,2,3,4])
            if(prob==1):
                get_varib = orig_stl[i]
                idx = get_varib.find("(")
                get,variable = get_varib[:idx],get_varib[idx:]
                get_varib = get[4:]
                if(get_varib in dict_of_func_id.keys()):
                    modified_stl[i] = dict_of_func_id[get_varib]
                    continue
                new_get = "get_"+generate_variable()
                rules = [f'the {get_varib} of X',f'X {get_varib}',f'{get_varib} property of X']
                lexicon_rule = random.choice(rules)+" => "+new_get+"(X)"
                custom_lexicon.append(lexicon_rule)
                modified_stl[i] = new_get+variable
                number_of_rules+=1

                dict_of_func_id[get_varib] = new_get+variable

        if("#VARIABLE" == template[i]):
            # prob = random.choice([j for j in range(1,21)])
            # if(prob==1):
            if(orig_stl[i] in varib_noun_dict):
                variable = orig_stl[i]
                lexicon_rule = f"{varib_noun_dict[variable]} => {variable}"
                # if(variable in dict_of_func_id.keys()):
                #     modified_stl[i] = dict_of_func_id[variable]
                #     continue
                # new_variable = generate_variable()
                # lexicon_rule = variable+" => "+new_variable
                custom_lexicon.append(lexicon_rule)
                # modified_stl[i] = new_variable
                number_of_rules+=1

                # dict_of_func_id[variable] = new_variable

        if ("#VARIABLE_BOUND" == template[i] and ">=" == template[i+1]):
            # prob = random.choice([j for j in range(1, 11)])
            # if (prob == 1):
            if(orig_stl[i] in varib_noun_dict):
                variable = orig_stl[i]
                lexicon_rule = f"{varib_noun_dict[variable]} => {variable}"
                # if(variable in dict_of_func_id.keys()):
                #     modified_stl[i] = dict_of_func_id[variable]
                #     modified_stl[i+4] = dict_of_func_id[variable]
                #     template[i+4]="#DONE"
                #     continue
                # new_variable = generate_variable()
                # lexicon_rule = variable + " => " + new_variable
                custom_lexicon.append(lexicon_rule)
                modified_stl[i] = variable

                modified_stl[i+4] = variable
                template[i+4]="#DONE"
                number_of_rules+=1

                dict_of_func_id[variable] = variable
    
    # #Include the synonyms as well
    # new_verbs_taken = []
    # for i in verbs_taken:
    #     new_verbs_taken.append(i)
    #     new_verbs_taken = new_verbs_taken+verbs_syn[i]
    
    # verbs_taken = new_verbs_taken

    # true_lexicon = custom_lexicon.copy()

    # if(number_of_rules>0 and number_of_rules<per_sample_lexicons):
    #     i = number_of_rules
    #     while(i<per_sample_lexicons):
    #         type_of_verb = random.sample(types_of_verbs,1)[0]
    #         verb = random.sample(verbs,1)[0]
    #         if(verb in verbs_taken):
    #             continue
    #         lexicon_rule,D,G = create_lexicon(type_of_verb,verb,train)
    #         custom_lexicon.append(lexicon_rule)
    #         i+=1

    random.shuffle(custom_lexicon)
    return custom_lexicon,modified_stl

if __name__=="__main__":
    v = random.choice(verbs)
    print(v)
    x = create_lexicon("#VERB4",v,True)
    print(x)



