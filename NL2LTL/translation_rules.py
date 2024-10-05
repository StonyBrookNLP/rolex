import numpy as np
import random
from list_of_funcs import verbs,verbs_type,verbs_syn,varib_noun_dict


def form_verb(verb,type="normal"):
    if(type=="normal"):
        return verb

    if(type=="singular"):
        return verbs_type[verb][0]


    if(type=="past_tense"):
        return verbs_type[verb][1]

translation_samples = {
    "#VARIABLE":['#VARIABLE', 'the #VARIABLE', 'the value of #VARIABLE', 'the value of variable #VARIABLE','a #VARIABLE','a value of #VARIABLE','a value of variable #VARIABLE'],

    "#NUMBER":['#NUMBER', 'the number #NUMBER', 'the value #NUMBER','a number #NUMBER','a value #NUMBER'],

    "#GET":['the #GET of #VARIABLE','#VARIABLE #GET','#GET property of #VARIABLE'],

    "IS":['is','was','will be','have been','must be'],

    "NOT_IS":['is not','was not','will not be','have not been','must not be'],

    "=":['','equal to', 'set to', 'equal to', 'settle to' ,'assigned as','assigned to'],

    ">":['above', 'over', 'more than', 'larger than','bigger than', 'higher than', 'greater than'],

    "<":['less than', 'smaller than','below', 'lower than'],

    ">=":['greater than or equal to', 'at least', 'no less than'],

    "<=":['less than or equal to', 'at most', 'no more than', 'no larger than'],

    "AND":['and',"and,"],

    "OR":["or","or,"],

    "IF":[
        'whenever',
        'in case that',
        'everytime when',
        'when',
        'if',
        'Globally, whenever',
        'Globally, in case that',
        'Globally, everytime when',
        'Globally, when',
        'Globally, if',
        'It is always the case that whenever',
        'It is always the case that in case that',
        'It is always the case that everytime when',
        'It is always the case that when',
        'It is always the case that if',
        'In the case',
        'On condition that',
        'As soon as'
    ],

    "THEN": [', then',
             ', then in response',
             ', then the following condition holds:',
             ', then the following condition is true:',
             ', then all of the following conditions hold:',
             ', then all of the following conditions are true:',
             ' then',
             ' then in response',
             ' then the following condition holds:',
             ' then the following condition is true:',
             ' then all of the following conditions hold:',
             ' then all of the following conditions are true:',
             ','
    ],

    "ADVERB":[
            'at once',
            'right away',
            'without any delay',
            'starting without any delay',
            'in no time',
            'at the same moment',
            'at the same time',
            'at the same time instant',
            'at the same time point'
    ],

    "NOT":[
        "not"
    ],

    "OTHERWISE":[
        'otherwise',
        'else'
    ],

    "AGENT":['client','server','system','filesystem'],

    "WITH":['with','with','with','with','with','with','with','with,','and'],

    "NEXT":['before','next']
}



# paraphrase_samples = {
#     "#VARIABLE":['#VARIABLE', 'the #VARIABLE', 'the value of #VARIABLE', 'the value of variable #VARIABLE','a #VARIABLE','a value of #VARIABLE','a value of variable #VARIABLE'],

#     "#NUMBER":['#NUMBER', 'the number #NUMBER', 'the value #NUMBER','a number #NUMBER','a value #NUMBER'],

#     "#GET":['the #GET of #VARIABLE','#VARIABLE #GET','#GET property of #VARIABLE'],

#     "IS":['is','was','will be','have been','must be'],

#     "NOT_IS":['is not','was not','will not be','have not been','must not be'],

#     "=":['equivalent to', 'set as', 'equated to', 'settle down to', 'assigned into'],

#     ">":['beyond', 'over', 'heavier than', 'upper than','huge than', 'much than'],

#     "<":['lesser than', 'smaller than','beneath', 'down than'],

#     ">=":['larger than or equal to', 'at the least most', 'no lower than'],

#     "<=":['lower than or equal to', 'at the highest', 'no bigger than', 'no greater than'],

#     "AND":['and',"and,"],

#     "OR":["or","or,"],

#     "IF":[
#         'globally, whenever',
#         'globally, in case that',
#         'globally, everytime when',
#         'globally, when',
#         'globally, if',
#         'it is always the case that whenever',
#         'it is always the case that in case that',
#         'it is always the case that everytime when',
#         'it is always the case that when',
#         'it is always the case that if',
#         'in the case',
#         'on condition that',
#         'as soon as'
#     ],

#     "THEN": [
#              ', then the following condition holds:',
#              ', then the following condition is true:',
#              ', then all of the following conditions hold:',
#              ', then all of the following conditions are true:',
#              ' then in response',
#              ' then the following condition holds:',
#              ' then the following condition is true:',
#              ' then all of the following conditions hold:',
#              ' then all of the following conditions are true:'
#     ],

#     "ADVERB":[
#             'at once',
#             'right away',
#             'without any delay',
#             'starting without any delay',
#             'in no time',
#             'at the same moment',
#             'at the same time',
#             'at the same time instant',
#             'at the same time point'
#     ],

#     "NOT":[
#         "not"
#     ],

#     "OTHERWISE":[
#         'otherwise',
#         'else'
#     ],

#     "AGENT":['client','server','system','filesystem'],

#     "WITH":['with','with','with','with','with','with','with','with,','and'],

#     "NEXT":['before','next']
# }

paraphrase_samples = translation_samples





def replace_name(scheme,identifier,name):
    for i in range(0,len(scheme)):
        scheme[i]=scheme[i].replace(identifier,name)

    return scheme


def next(arr,idx,item):
    for i in range(idx,len(arr)):
        if(arr[i]==item):
            return i
    return -1

def extract_verb(phrase):
    for i in range(0,len(phrase)):
        if(phrase[i]=="_"):
            return phrase[i+1:]

def pick_synonym_verb(verb):
    return random.choice(verbs_syn[verb])

def sample_translation(translation_scheme,template):
    translation = ""

    for i in range(0,len(translation_scheme)):
        temp = translation_scheme[i]
        sample = random.sample(temp,1)
        translation+=sample[0]+" "

    temp = translation.split()

    if(temp[0]=="the" or temp[0] in translation_samples["IF"] or temp[0] in translation_samples["AGENT"]):
        translation = translation[0].capitalize()+translation[1:]+"."
    else:
        translation = translation+"."
    
    return translation


def generate_translate_scheme(orig_template,orig_stl,is_train):
    translation_scheme = []
    template = orig_template.copy()
    stl = orig_stl.copy()
    negate_flag = False
    var_bound = False
    i = 0

    while(i<len(template)):

        if(template[i]=="#VARIABLE" or template[i]=="#VARIABLE_BOUND"):
            if(template[i]=="#VARIABLE_BOUND"):
                var_bound = True
            var_name = stl[i]
            if(var_name in varib_noun_dict):
                var_name = varib_noun_dict[var_name]
            scheme = translation_samples["#VARIABLE"].copy()
            scheme = replace_name(scheme,"#VARIABLE",var_name)
            translation_scheme.append(scheme)

        if(template[i]=="#NUMBER"):
            num = stl[i]
            scheme = translation_samples["#NUMBER"].copy()
            scheme = replace_name(scheme,"#NUMBER",num)
            translation_scheme.append(scheme)

        if(template[i]=="#OPERATION"):
            if negate_flag:
                scheme = translation_samples["NOT_IS"].copy()
                negate_flag = False
            else:
                scheme = translation_samples["IS"].copy()
            translation_scheme.append(scheme)

            op = stl[i]
            scheme = translation_samples[op].copy()
            translation_scheme.append(scheme)

        if("#VERB" in template[i]):

            if(template[i]=="#VERB0"):
                verb = stl[i]
                scheme = translation_samples["AGENT"].copy()
                translation_scheme.append(scheme)
                scheme = [form_verb(verb,"singular")]

                if(is_train):
                    synonyms = verbs_syn[verb]
                    for j in synonyms:
                        scheme.append(form_verb(j,"singular"))
                
                translation_scheme.append(scheme)

            elif(template[i]=="#VERB1"):
                verb = stl[i]
                idx = next(template,i+1,")")
                idx = next(template,idx+1,")")
                template.insert(idx,"#VERB_"+verb)
                stl.insert(idx,"#DUMMY")
                i = i + 5
                continue

            elif (template[i] == "#VERB2"):
                verb = stl[i]
                idx = next(template, i+1, "WITH")
                idx = next(template, idx+1, "WITH")
                template.insert(idx,"#VERB_"+verb)
                stl.insert(idx, "#DUMMY")
                i = i + 5
                continue

            elif (template[i] == "#VERB3"):
                verb = stl[i]
                idx = next(template, i, "WITH")
                idx = next(template, idx+1, "WITH")
                template.insert(idx,"#VERB_"+verb)
                stl.insert(idx, "#DUMMY")
                i = i + 5
                continue

            elif (template[i] == "#VERB4"):
                verb = stl[i]
                idx = next(template, i, "WITH")
                idx = next(template, idx+1, "WITH")
                template.insert(idx,"#VERB_"+verb)
                stl.insert(idx, "#DUMMY")
                i = i + 5
                continue

            else:
                scheme = translation_samples["IS"].copy()
                translation_scheme.append(scheme)
                verb = extract_verb(template[i])
                scheme = [form_verb(verb, "past_tense")]
                
                if(is_train):
                    synonyms = verbs_syn[verb]
                    for j in synonyms:
                        scheme.append(form_verb(j, "past_tense"))

                translation_scheme.append(scheme)
        
        if(template[i]=="#GET"):
            get_var = stl[i]
            idx1,idx2,idx3 = get_var.find("_"),get_var.find("("),get_var.find(")")
            get = get_var[idx1+1:idx2]
            var = get_var[idx2+1:idx3]

            if get in varib_noun_dict.keys():
                get = varib_noun_dict[get]
            
            if var in varib_noun_dict.keys():
                var = varib_noun_dict[var]
              
            scheme = translation_samples["#GET"].copy()
            scheme = replace_name(scheme,"#GET",get)
            scheme = replace_name(scheme,"#VARIABLE",var)
            translation_scheme.append(scheme)

        if(template[i]=="AND"):
            scheme = translation_samples["AND"].copy()
            translation_scheme.append(scheme)

        if(template[i]=="OR"):
            scheme = translation_samples["OR"].copy()
            translation_scheme.append(scheme)
        
        if(template[i]=="NEXT"):
            scheme = translation_samples["NEXT"].copy()
            translation_scheme.append(scheme)

        if(template[i]=="WITH"):
            scheme = translation_samples["WITH"].copy()
            translation_scheme.append(scheme)

        if(template[i]=="->"):
            if(translation_scheme[0]==translation_samples["IF"]):
                scheme = translation_samples["OTHERWISE"].copy()
                translation_scheme.pop(-1)
                translation_scheme.append(scheme)
                template[i+3]="_"
            else:
                scheme_1 = translation_samples["IF"].copy()
                scheme_2 = translation_samples["THEN"].copy()
                translation_scheme.insert(0,scheme_1)
                translation_scheme.append(scheme_2)

        if(template[i]=="NOT"):
            negate_flag = True

        if(template[i]==">="):
            if var_bound :
                if negate_flag:
                    scheme = translation_samples["NOT_IS"].copy()
                    translation_scheme.append(scheme)
                    scheme = translation_samples[">="][:2].copy()
                    translation_scheme.append(scheme)
                    template[i+2] = "OR"
                else:
                    scheme = translation_samples["IS"].copy()
                    translation_scheme.append(scheme)
                    scheme = translation_samples[">="].copy()
                    translation_scheme.append(scheme)
                    
                var_bound = False
                template[i+3]="_"
            else:
                if negate_flag:
                    scheme = translation_samples["NOT_IS"].copy()
                    negate_flag = False
                    translation_scheme.append(scheme)
                    scheme = translation_samples[">="][:2].copy()
                    translation_scheme.append(scheme)
                else:
                    scheme = translation_samples["IS"].copy()
                    translation_scheme.append(scheme)
                    scheme = translation_samples[">="].copy()
                    translation_scheme.append(scheme)

        if(template[i]=="<="):
            if negate_flag:
                scheme = translation_samples["NOT_IS"].copy()
                negate_flag = False
                translation_scheme.append(scheme)
                scheme = translation_samples["<="][:2].copy()
                translation_scheme.append(scheme)
            else:
                scheme = translation_samples["IS"].copy()
                translation_scheme.append(scheme)
                scheme = translation_samples["<="].copy()
                translation_scheme.append(scheme)
        
        if(template[i]==">"):
            if negate_flag:
                scheme = translation_samples["NOT_IS"].copy()
                negate_flag = False
                translation_scheme.append(scheme)
                scheme = translation_samples[">"][:2].copy()
                translation_scheme.append(scheme)
            else:
                scheme = translation_samples["IS"].copy()
                translation_scheme.append(scheme)
                scheme = translation_samples[">"].copy()
                translation_scheme.append(scheme)

        if(template[i]=="<"):
            if negate_flag:
                scheme = translation_samples["NOT_IS"].copy()
                negate_flag = False
                translation_scheme.append(scheme)
                scheme = translation_samples["<"][:2].copy()
                translation_scheme.append(scheme)
            else:
                scheme = translation_samples["IS"].copy()
                translation_scheme.append(scheme)
                scheme = translation_samples["<"].copy()
                translation_scheme.append(scheme)

        i+=1

    return translation_scheme

def paraphrase_if(translation_scheme):

    for i in range(0,len(translation_scheme)):
        if(translation_scheme[i]==translation_samples["THEN"]):
            break

    new_translation_scheme = translation_scheme[i+1:]+[translation_samples["IF"]]+translation_scheme[:i]

    return new_translation_scheme





def generate_paraphrase_scheme(orig_template,orig_stl):
    translation_scheme = []
    template = orig_template.copy()
    stl = orig_stl.copy()
    negate_flag = False
    var_bound = False
    i = 0

    while(i<len(template)):

        if(template[i]=="#VARIABLE" or template[i]=="#VARIABLE_BOUND"):
            if(template[i]=="#VARIABLE_BOUND"):
                var_bound = True
            var_name = stl[i]
            #scheme = paraphrase_samples["#VARIABLE"].copy()
            if(var_name in varib_noun_dict):
                var_name = varib_noun_dict[var_name]
            scheme = translation_samples["#VARIABLE"].copy()
            scheme = replace_name(scheme,"#VARIABLE",var_name)
            translation_scheme.append(scheme)

        if(template[i]=="#NUMBER"):
            num = stl[i]
            #scheme = paraphrase_samples["#NUMBER"].copy()
            scheme = translation_samples["#NUMBER"].copy()
            scheme = replace_name(scheme,"#NUMBER",num)
            translation_scheme.append(scheme)

        if(template[i]=="#OPERATION"):
            if negate_flag:
                #scheme = paraphrase_samples["NOT_IS"].copy()
                scheme = translation_samples["NOT_IS"].copy()
                negate_flag = False
            else:
                #scheme = paraphrase_samples["IS"].copy()
                scheme = translation_samples["IS"].copy()
            translation_scheme.append(scheme)

            op = stl[i]
            #scheme = paraphrase_samples[op].copy()
            scheme = translation_samples[op].copy()
            translation_scheme.append(scheme)

        if("#VERB" in template[i]):
            if(template[i]=="#VERB0"):
                verb = stl[i]
                verb = random.choice(verbs_syn[verb])
                scheme = paraphrase_samples["AGENT"].copy()
                translation_scheme.append(scheme)
                scheme = [form_verb(verb,"singular")]

                # synonyms = verbs_syn[verb]
                # for j in synonyms:
                #     scheme.append(form_verb(j,"singular"))

                translation_scheme.append(scheme)

            elif(template[i]=="#VERB1"):
                verb = stl[i]
                idx = next(template,i+1,")")
                idx = next(template,idx+1,")")
                template.insert(idx,"#VERB_"+verb)
                stl.insert(idx,"#DUMMY")
                i = i + 5
                continue

            elif (template[i] == "#VERB2"):
                verb = stl[i]
                idx = next(template, i+1, "WITH")
                idx = next(template, idx+1, "WITH")
                template.insert(idx,"#VERB_"+verb)
                stl.insert(idx, "#DUMMY")
                i = i + 5
                continue

            elif (template[i] == "#VERB3"):
                verb = stl[i]
                idx = next(template, i, "WITH")
                idx = next(template, idx+1, "WITH")
                template.insert(idx,"#VERB_"+verb)
                stl.insert(idx, "#DUMMY")
                i = i + 5
                continue

            elif (template[i] == "#VERB4"):
                verb = stl[i]
                idx = next(template, i, "WITH")
                idx = next(template, idx+1, "WITH")
                template.insert(idx,"#VERB_"+verb)
                stl.insert(idx, "#DUMMY")
                i = i + 5
                continue

            else:
                scheme = paraphrase_samples["IS"].copy()
                translation_scheme.append(scheme)
                verb = extract_verb(template[i])
                verb = random.choice(verbs_syn[verb])
                scheme = [form_verb(verb, "past_tense")]

                # synonyms = verbs_syn[verb]
                # for j in synonyms:
                #     scheme.append(form_verb(j,"past_tense"))
                
                translation_scheme.append(scheme)

        if(template[i]=="#GET"):
            get_var = stl[i]
            idx1,idx2,idx3 = get_var.find("_"),get_var.find("("),get_var.find(")")
            get = get_var[idx1+1:idx2]
            var = get_var[idx2+1:idx3]

            if get in varib_noun_dict.keys():
                get = varib_noun_dict[get]
            
            if var in varib_noun_dict.keys():
                var = varib_noun_dict[var]
            
            #scheme = paraphrase_samples["#GET"].copy()
            scheme = translation_samples["#GET"].copy()
            scheme = replace_name(scheme,"#GET",get)
            scheme = replace_name(scheme,"#VARIABLE",var)
            translation_scheme.append(scheme)

        if(template[i]=="AND"):
            #scheme = paraphrase_samples["AND"].copy()
            scheme = translation_samples["AND"].copy()
            translation_scheme.append(scheme)

        if(template[i]=="OR"):
            #scheme = paraphrase_samples["OR"].copy()
            scheme = translation_samples["OR"].copy()
            translation_scheme.append(scheme)
        
        if(template[i]=="NEXT"):
            #scheme = paraphrase_samples["NEXT"].copy()
            scheme = translation_samples["NEXT"].copy()
            translation_scheme.append(scheme)

        if(template[i]=="WITH"):
            #scheme = paraphrase_samples["WITH"].copy()
            scheme = translation_samples["WITH"].copy()
            translation_scheme.append(scheme)

        if(template[i]=="->"):
            if(translation_scheme[0]==paraphrase_samples["IF"]):
                #scheme = paraphrase_samples["OTHERWISE"].copy()
                scheme = translation_samples["OTHERWISE"].copy()
                translation_scheme.pop(-1)
                translation_scheme.append(scheme)
            else:
                # scheme_1 = paraphrase_samples["IF"].copy()
                # scheme_2 = paraphrase_samples["THEN"].copy()
                scheme_1 = translation_samples["IF"].copy()
                scheme_2 = translation_samples["THEN"].copy()
                translation_scheme.insert(0,scheme_1)
                translation_scheme.append(scheme_2)

        if(template[i]=="NOT"):
            negate_flag = True

        if(template[i]==">="):
            if var_bound :
                if negate_flag:
                    #scheme = paraphrase_samples["NOT_IS"].copy()
                    scheme = translation_samples["NOT_IS"].copy()
                    translation_scheme.append(scheme)
                    #scheme = paraphrase_samples[">="][:2].copy()
                    scheme = translation_samples[">="][:2].copy()
                    translation_scheme.append(scheme)
                    template[i+2] = "OR"
                else:
                    #scheme = paraphrase_samples["IS"].copy()
                    scheme = translation_samples["IS"].copy()
                    translation_scheme.append(scheme)
                    #scheme = paraphrase_samples[">="].copy()
                    scheme = translation_samples[">="].copy()
                    translation_scheme.append(scheme)
                var_bound = False
                template[i+3]="_"
            else:
                if negate_flag:
                    #scheme = paraphrase_samples["NOT_IS"].copy()
                    scheme = translation_samples["NOT_IS"].copy()
                    negate_flag = False
                    translation_scheme.append(scheme)
                    #scheme = paraphrase_samples[">="][:2].copy()
                    scheme = translation_samples[">="][:2].copy()
                    translation_scheme.append(scheme)
                else:
                    #scheme = paraphrase_samples["IS"].copy()
                    scheme = translation_samples["IS"].copy()
                    translation_scheme.append(scheme)
                    #scheme = paraphrase_samples[">="].copy()
                    scheme = translation_samples[">="].copy()
                    translation_scheme.append(scheme)

        if(template[i]=="<="):
            if negate_flag:
                #scheme = paraphrase_samples["NOT_IS"].copy()
                scheme = translation_samples["NOT_IS"].copy()
                negate_flag = False
                translation_scheme.append(scheme)
                #scheme = paraphrase_samples["<="][:2].copy()
                scheme = translation_samples["<="][:2].copy()
                translation_scheme.append(scheme)
            else:
                #scheme = paraphrase_samples["IS"].copy()
                scheme = translation_samples["IS"].copy()
                translation_scheme.append(scheme)
                #scheme = paraphrase_samples["<="].copy()
                scheme = translation_samples["<="].copy()
                translation_scheme.append(scheme)
        

        if(template[i]==">"):
            if negate_flag:
                scheme = translation_samples["NOT_IS"].copy()
                negate_flag = False
                translation_scheme.append(scheme)
                scheme = translation_samples[">"][:2].copy()
                translation_scheme.append(scheme)
            else:
                scheme = translation_samples["IS"].copy()
                translation_scheme.append(scheme)
                scheme = translation_samples[">"].copy()
                translation_scheme.append(scheme)

        if(template[i]=="<"):
            if negate_flag:
                scheme = translation_samples["NOT_IS"].copy()
                negate_flag = False
                translation_scheme.append(scheme)
                scheme = translation_samples["<"][:2].copy()
                translation_scheme.append(scheme)
            else:
                scheme = translation_samples["IS"].copy()
                translation_scheme.append(scheme)
                scheme = translation_samples["<"].copy()
                translation_scheme.append(scheme)

        i+=1

    return translation_scheme