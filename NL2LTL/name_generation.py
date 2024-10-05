import numpy as np
import random
from list_of_funcs import verbs,nouns,varib_noun_dict,noun_varib_dict

#Numerical values are generated between -range and range
limit = 1000
small_letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
capital_letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
digits = ['0','1','2','3','4','5','6','7','8','9']
symbols = ["_","-"]
ops = ["=","<",">","<=",">="]

verb_length = len(verbs)
noun_length = len(nouns)

all_letters = small_letters+capital_letters
all_characters = small_letters+capital_letters+digits+symbols

def generate_variable_symbol():
    prob = random.choice([0, 1, 2, 3])
    if (prob == 1):
        var_name = "i"
    else:
        var_name = random.choice(all_letters)

    size = np.random.randint(10)

    for i in range(size):
        var_name += random.choice(all_characters)

    return var_name


def generate_variable(construct_space):
    prob = random.choice([i for i in range(0,20)])
    if(prob==1):
        if(construct_space==1):
            possible_nouns = nouns[:len(nouns)//2]
            possible_varibs = [noun_varib_dict[i] for i in possible_nouns]
            return random.choice(possible_varibs)
        else:
            possible_nouns = nouns[len(nouns)//2:]
            possible_varibs = [noun_varib_dict[i] for i in possible_nouns]
            return random.choice(possible_varibs)
    else:
        var = generate_variable_symbol()
        while(var=="a" or var=="the" or len(var)<3):
            var = generate_variable_symbol()
        return var

def get_random_numerical_value():
    prob = random.choice([0,1,2,3])

    if(prob==1):
        return np.random.randint(limit)
    elif(prob==2):
        return -np.random.randint(limit)
    elif(prob==3):
        return np.random.randint(limit*100)/100
    else:
        return -np.random.randint(limit*100)/100


def generate_single_number():
    return str(get_random_numerical_value())

def generate_bounded_number():
    first = get_random_numerical_value()
    second = first-1
    while(second<first):
        second = get_random_numerical_value()
    return str(first),str(second)

def generate_get_noun(construct_space):
    prob = random.choice([0,1])
    if(prob==0):
        if(construct_space==1):
            return "get_"+random.choice(nouns[:noun_length//2])+"("+generate_variable(construct_space)+")"
        elif(construct_space==2):
            return "get_"+random.choice(nouns[noun_length//2:])+"("+generate_variable(construct_space)+")"
        else:
            return "get_"+random.choice(nouns)+"("+generate_variable(construct_space)+")"
    else:
        return "get_"+generate_variable(construct_space)+"("+generate_variable(construct_space)+")"

def generate_function_verb(construct_space):
    if(construct_space==1):
        return random.choice(verbs[:verb_length//2])
    elif(construct_space==2):
        return random.choice(verbs[verb_length//2:])
    else:
        return random.choice(verbs)
    
def generate_lexicon_noun(construct_space):
    if(construct_space==1):
        search_space = nouns[:noun_length//2]
    else:
        search_space = nouns[noun_length//2:]
    noun_selected = random.choice(search_space)
    return noun_selected,noun_varib_dict[noun_selected]

def generate_entity(id):
    return "E"

def generate_operation():
    return random.choice(ops)




if __name__ == "__main__":
    print(generate_variable())
    print(generate_single_number())
    print(generate_bounded_number())
    print(generate_get_noun())
    print(generate_operation())