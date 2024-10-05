import numpy as np
import random
from name_generation import generate_variable,generate_single_number,generate_bounded_number,generate_entity,generate_get_noun,generate_function_verb,generate_operation
from auxillary_funcs import find
from translation_rules import translation_samples

#np.random.seed(1245)
#random.seed(1245)

class stl_rules:
    def __init__(self,construct_space):
        self.rules_dict = {}
        self.create_rules_dictionary()
        self.index_count_dict = {
            "#VARIABLE": 0,
            "#NUMBER": 0,
            "#VERB": 0,
            "#GET": 0,
            "#ENTITY": 0
        }

        self.list_of_terminals ={
            "ALWAYS":True,
            "(":True,
            ")":True,
            "#VARIABLE":True,
            "#VARIABLE_BOUND":True,
            "#NUMBER":True,
            "#VERB":True,
            "#VERB0":True,
            "#VERB1":True,
            "#VERB2":True,
            "#VERB3":True,
            "#VERB4":True,
            "#GET":True,
            "#ENTITY":True,
            "#OPERATION": True,
            "=":True,
            ">=":True,
            ">":True,
            "<":True,
            "<=":True,
            "AND":True,
            "OR":True,
            "->":True,
            ".":True,
            "NOT":True,
            "WITH":True,
            "NEXT":True,
            "#NOUN":True
        }

        self.list_of_non_terminals={
            "start":True,
            "simple_phrase":True,
            "alpha":True,
            "predicate":True,
            "function": True,
            "bounded_predicate":True,
            "connective":True,
        }

        self.construct_space = construct_space

    def create_rules_dictionary(self):
        self.rules_dict["start"] = {
            "id":[0,1,2],
            "substitutions":[
                ["ALWAYS","(","simple_phrase",")"],
                ["ALWAYS","(","simple_phrase","->","simple_phrase",")"],
                ["ALWAYS", "(", "simple_phrase", "->", "simple_phrase","OR","->","simple_phrase", ")"]
            ],
            "probability":[1/3,1/3,1/3]
            #"probability":[0,0,1]
        }

        self.rules_dict["simple_phrase"] = {
            "id":[3,4,5],
            "substitutions":[
                ["alpha"],
                ["alpha","connective","alpha"],
                ["alpha","connective","alpha","connective","alpha"]
            ],
            "probability":[5/8,2/8,1/8]
        }

        self.rules_dict["alpha"] = {
            "id":[6,7,8],
            "substitutions": [
                ["(","predicate",")"],
                ["(","bounded_predicate",")"],
                ["(","function",")"]
            ],
            "probability": [0.3,0.1,0.6]
        }

        self.rules_dict["predicate"]={
            "id":[9,10,11,12,13,14,15,16,17,18,19,20],
            "substitutions":[
                ["#VARIABLE","#OPERATION","#NUMBER"],
                ["#VARIABLE","#OPERATION","#VARIABLE"],
                ["#GET","#OPERATION","#NUMBER"],
                ["#GET","#OPERATION","#VARIABLE"],
                ["#VARIABLE","#OPERATION","#GET"],
                ["#VARIABLE"],
                ["NOT","(","#VARIABLE", "#OPERATION", "#NUMBER",")"],
                ["NOT","(","#VARIABLE", "#OPERATION", "#VARIABLE",")"],
                ["NOT","(","#GET", "#OPERATION", "#NUMBER",")"],
                ["NOT","(","#GET", "#OPERATION", "#VARIABLE",")"],
                ["NOT","(","#VARIABLE", "#OPERATION", "#GET",")"],
                ["NOT","#VARIABLE"],
            ],
            "probability":[2/12,1/12,1/12,1/12,1/12,1/12,1/12,1/12,1/12,1/12,1/12,0]
        }

        self.rules_dict["bounded_predicate"]={
            "id":[21,22,23,24],
            "substitutions": [
                ["#VARIABLE_BOUND", ">=", "#NUMBER","AND","#VARIABLE_BOUND","<=","#NUMBER"],
                ["#VARIABLE_BOUND", ">=", "#VARIABLE","AND","#VARIABLE_BOUND","<=","#VARIABLE"],
                # ["NOT","(","#VARIABLE_BOUND", ">=", "#NUMBER", "AND", "#VARIABLE_BOUND", "<=", "#NUMBER",")"],
                # ["NOT","(","#VARIABLE_BOUND", ">=", "#VARIABLE", "AND", "#VARIABLE_BOUND", "<=", "#VARIABLE",")"],
                ["#VARIABLE_BOUND", "<", "#NUMBER", "OR", "#VARIABLE_BOUND", ">", "#NUMBER"],
                ["#VARIABLE_BOUND", "<", "#VARIABLE", "OR", "#VARIABLE_BOUND", ">", "#VARIABLE"]
            ],
            "probability":[1/4,1/4,1/4,1/4]
        }

        self.rules_dict["function"]={
            "id":[25,26,27,28,29],
            "substitutions":[
                ["#VERB0", "(", "#ENTITY", ")"],
                ["#VERB1","(","#ENTITY",")","WITH","#ENTITY",".","predicate"],
                ["#VERB2", "(", "#ENTITY", ")", "WITH", "#ENTITY", ".", "predicate", "WITH", "#ENTITY", ".", "predicate"],
                ["#VERB3", "(", "#ENTITY", ")", "WITH", "#ENTITY", ".", "predicate", "WITH", "#ENTITY", ".", "predicate", "WITH", "#ENTITY", ".", "predicate"],
                ["#VERB4", "(", "#ENTITY", ")", "WITH", "#ENTITY", ".", "predicate", "WITH", "#ENTITY", ".", "predicate", "WITH", "#ENTITY", ".", "predicate", "WITH", "#ENTITY", ".", "predicate"]
            ],
            "probability":[0.2,0.4,0.2,0.1,0.1]
        }


        self.rules_dict["connective"]={
            "id":[30,31,32],
            "substitutions":[
                ["AND"],
                ["OR"],
                ["NEXT"]
            ],
            #"probability":[19/40,19/40,1/20]
            "probability":[1/3,1/3,1/3]
        }

    def get_index(self,terminal):
        idx = self.index_count_dict[terminal]
        self.index_count_dict[terminal]+=1
        return idx

    def transform(self,next_template_parse,rule_id):
        next_stl_parse = next_template_parse.copy()

        if(rule_id==9):
            var = generate_variable(self.construct_space)
            number = generate_single_number()
            op = generate_operation()

            next_stl_parse[0]=var
            next_stl_parse[1]=op
            next_stl_parse[2]=number

        if(rule_id==10):
            var_1 = generate_variable(self.construct_space)
            var_2 = generate_variable(self.construct_space)
            op = generate_operation()

            next_stl_parse[0] = var_1
            next_stl_parse[1] = op
            next_stl_parse[2] = var_2

        if(rule_id==11):
            get = generate_get_noun(self.construct_space)
            num = generate_single_number()
            op = generate_operation()

            next_stl_parse[0]=get
            next_stl_parse[1]=op
            next_stl_parse[2]=num

        if(rule_id==12):
            get = generate_get_noun(self.construct_space)
            var = generate_variable(self.construct_space)
            op = generate_operation()

            next_stl_parse[0] = get
            next_stl_parse[1] = op
            next_stl_parse[2] = var

        if(rule_id==13):
            get = generate_get_noun(self.construct_space)
            var = generate_variable(self.construct_space)
            op = generate_operation()

            next_stl_parse[0] = var
            next_stl_parse[1] = op
            next_stl_parse[2] = get

        if(rule_id==14):
            var = generate_variable(self.construct_space)

            next_stl_parse[0] = var

        if(rule_id==15):
            var = generate_variable(self.construct_space)
            number = generate_single_number()
            op = generate_operation()

            next_stl_parse[2] = var
            next_stl_parse[3] = op
            next_stl_parse[4] = number

        if(rule_id==16):
            var_1 = generate_variable(self.construct_space)
            var_2 = generate_variable(self.construct_space)
            op = generate_operation()

            next_stl_parse[2] = var_1
            next_stl_parse[3] = op
            next_stl_parse[4] = var_2

        if(rule_id==17):
            get = generate_get_noun(self.construct_space)
            num = generate_single_number()
            op = generate_operation()

            next_stl_parse[2] = get
            next_stl_parse[3] = op
            next_stl_parse[4] = num

        if(rule_id==18):
            get = generate_get_noun(self.construct_space)
            var = generate_variable(self.construct_space)
            op = generate_operation()

            next_stl_parse[2] = get
            next_stl_parse[3] = op
            next_stl_parse[4] = var

        if(rule_id==19):
            get = generate_get_noun(self.construct_space)
            var = generate_variable(self.construct_space)
            op = generate_operation()

            next_stl_parse[2] = var
            next_stl_parse[3] = op
            next_stl_parse[4] = get

        if(rule_id==20):
            var = generate_variable(self.construct_space)

            next_stl_parse[1] = var

        if (rule_id == 21):
            var = generate_variable(self.construct_space)
            num1, num2 = generate_bounded_number()

            next_stl_parse[0] = var
            next_stl_parse[1] = ">="
            next_stl_parse[2] = num1
            next_stl_parse[4] = var
            next_stl_parse[5] = "<="
            next_stl_parse[6] = num2

        if (rule_id == 22):
            var1 = generate_variable(self.construct_space)
            var2 = generate_variable(self.construct_space)
            var3 = generate_variable(self.construct_space)

            next_stl_parse[0] = var1
            next_stl_parse[1] = ">="
            next_stl_parse[2] = var2
            next_stl_parse[4] = var1
            next_stl_parse[5] = "<="
            next_stl_parse[6] = var3

        # if (rule_id == 23):
        #     var = generate_variable(self.construct_space)
        #     num1, num2 = generate_bounded_number()

        #     next_stl_parse[2] = var
        #     next_stl_parse[3] = ">="
        #     next_stl_parse[4] = num1
        #     next_stl_parse[6] = var
        #     next_stl_parse[7] = "<="
        #     next_stl_parse[8] = num2

        # if (rule_id == 24):
        #     var1 = generate_variable(self.construct_space)
        #     var2 = generate_variable(self.construct_space)
        #     var3 = generate_variable(self.construct_space)

        #     next_stl_parse[2] = var1
        #     next_stl_parse[3] = ">="
        #     next_stl_parse[4] = var2
        #     next_stl_parse[6] = var1
        #     next_stl_parse[7] = "<="
        #     next_stl_parse[8] = var3

        if (rule_id == 23):
            var = generate_variable(self.construct_space)
            num1, num2 = generate_bounded_number()

            next_stl_parse[0] = var
            next_stl_parse[1] = "<"
            next_stl_parse[2] = num1
            next_stl_parse[4] = var
            next_stl_parse[5] = ">"
            next_stl_parse[6] = num2

        if (rule_id == 24):
            var1 = generate_variable(self.construct_space)
            var2 = generate_variable(self.construct_space)
            var3 = generate_variable(self.construct_space)

            next_stl_parse[0] = var1
            next_stl_parse[1] = "<"
            next_stl_parse[2] = var2
            next_stl_parse[4] = var1
            next_stl_parse[5] = ">"
            next_stl_parse[6] = var3

        if(rule_id==25 or rule_id==26 or rule_id==27 or rule_id==28 or rule_id==29):
            verb = generate_function_verb(self.construct_space)
            entity = generate_entity(self.get_index("#ENTITY"))

            next_stl_parse[0]=verb

            for i in range(1,len(next_stl_parse)):
                if(next_stl_parse[i]=="#ENTITY"):
                    next_stl_parse[i]=entity

        return next_stl_parse

    def handle_otherwise(self,template,stl):
        implication_idx = find(template,"->")

        simple_phrase_template = ["NOT","("]+template[2:implication_idx[0]]+[")"]
        template = template[0:implication_idx[1]]+simple_phrase_template+template[implication_idx[1]:]

        simple_phrase_stl = ["NOT","("]+stl[2:implication_idx[0]]+[")"]
        stl = stl[0:implication_idx[1]]+simple_phrase_stl+stl[implication_idx[1]:]

        return template,stl



    def generate_next_parse(self,node):
        subs = self.rules_dict[node]["substitutions"]
        probs = self.rules_dict[node]["probability"]
        id = self.rules_dict[node]["id"]

        index = np.random.choice(len(subs),size=1,p=probs)[0]

        next_template_parse = subs[index]
        rule_id = id[index]
        next_stl_parse = self.transform(next_template_parse,rule_id)

        return next_template_parse,next_stl_parse,rule_id