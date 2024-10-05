import numpy as np
import pandas as pd
import random
from list_of_funcs import verbs,nouns,noun_varib_dict
from translation_rules import form_verb


def create_lexicons(verbs,nouns):
    nl = []
    ltl = []
    pk = []

    for verb in verbs:
        #Add verb_0
        nl_temp = f"{form_verb(verb,'singular')}"
        ltl_temp = f"{verb} ( E )"

        nl.append(nl_temp)
        ltl.append(ltl_temp)
        pk.append(nl_temp+" => "+ltl_temp)

        #Add verb_1
        nl_temp = f"A is {form_verb(verb,'past_tense')}"
        ltl_temp = f"{verb} ( E ) WITH E . A"

        nl.append(nl_temp)
        ltl.append(ltl_temp)
        pk.append(nl_temp+" => "+ltl_temp)


        #Add verb_2
        nl_temp = f"A is {form_verb(verb, 'past_tense')} with B"
        ltl_temp = f"{verb} ( E ) WITH E . A WITH E . B"

        nl.append(nl_temp)
        ltl.append(ltl_temp)
        pk.append(nl_temp+" => "+ltl_temp)

        #Add verb_3
        nl_temp = f"A is {form_verb(verb, 'past_tense')} with B and C"
        ltl_temp = f"{verb} ( E ) WITH E . A WITH E . B WITH E . C"

        nl.append(nl_temp)
        ltl.append(ltl_temp)
        pk.append(nl_temp+" => "+ltl_temp)

        #Add verb_4
        nl_temp = f"A is {form_verb(verb, 'past_tense')} with B and C and D"
        ltl_temp = f"{verb} ( E ) WITH E . A WITH E . B WITH E . C WITH E . D"

        nl.append(nl_temp)
        ltl.append(ltl_temp)
        pk.append(nl_temp+" => "+ltl_temp)
    
    for noun in nouns:
        #Add the nouns
        nl_temp = noun
        ltl_temp = noun_varib_dict[noun]

        nl.append(nl_temp)
        ltl.append(ltl_temp)
        pk.append(nl_temp+" => "+ltl_temp)

    
    return nl,ltl,pk




def create_lexicons_rnf(verbs,nouns,all_verbs):
    nl = []
    ltl = []
    pk = []

    

    for verb in verbs:
        func_id = 5*all_verbs.index(verb)

        #Add verb_0
        nl_temp = f"{form_verb(verb,'singular')}"
        ltl_temp = f"func{func_id} ( E )"

        nl.append(nl_temp)
        ltl.append(ltl_temp)
        pk.append(nl_temp+" => "+ltl_temp)

        #Add verb_1
        nl_temp = f"A is {form_verb(verb,'past_tense')}"
        ltl_temp = f"func{func_id+1} ( E ) WITH E . A"

        nl.append(nl_temp)
        ltl.append(ltl_temp)
        pk.append(nl_temp+" => "+ltl_temp)


        #Add verb_2
        nl_temp = f"A is {form_verb(verb, 'past_tense')} with B"
        ltl_temp = f"func{func_id+2} ( E ) WITH E . A WITH E . B"

        nl.append(nl_temp)
        ltl.append(ltl_temp)
        pk.append(nl_temp+" => "+ltl_temp)

        #Add verb_3
        nl_temp = f"A is {form_verb(verb, 'past_tense')} with B and C"
        ltl_temp = f"func{func_id+3} ( E ) WITH E . A WITH E . B WITH E . C"

        nl.append(nl_temp)
        ltl.append(ltl_temp)
        pk.append(nl_temp+" => "+ltl_temp)

        #Add verb_4
        nl_temp = f"A is {form_verb(verb, 'past_tense')} with B and C and D"
        ltl_temp = f"func{func_id+4} ( E ) WITH E . A WITH E . B WITH E . C WITH E . D"

        nl.append(nl_temp)
        ltl.append(ltl_temp)
        pk.append(nl_temp+" => "+ltl_temp)
    
    for noun in nouns:
        #Add the nouns
        nl_temp = noun
        ltl_temp = noun_varib_dict[noun]

        nl.append(nl_temp)
        ltl.append(ltl_temp)
        pk.append(nl_temp+" => "+ltl_temp)

    
    return nl,ltl,pk

def save_as_dataframe(nl,ltl,pk,filename):
    df = pd.DataFrame({"nl":nl,"ltl":ltl,"pk":pk})
    df.to_csv(filename)


if __name__ == "__main__":
    verb_div = len(verbs)//2
    noun_div = len(nouns)//2

    indist_nl,indist_ltl,indist_pk = create_lexicons(verbs[:verb_div],nouns[:noun_div])
    outdist_nl,outdist_ltl,outdist_pk = create_lexicons(verbs[verb_div:],nouns[noun_div:])
    outdist_rnf_nl,outdist_rnf_ltl,outdist_rnf_pk = create_lexicons_rnf(verbs[verb_div:],nouns[noun_div:],verbs)

    save_as_dataframe(indist_nl,indist_ltl,indist_pk,"dataset/train_pk.csv")
    save_as_dataframe(outdist_nl,outdist_ltl,outdist_pk,"dataset/test_pk.csv")
    save_as_dataframe(outdist_rnf_nl,outdist_rnf_ltl,outdist_rnf_pk,"dataset/test_rnf_pk.csv")

    