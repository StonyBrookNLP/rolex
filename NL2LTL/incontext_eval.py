import backoff
import time
import pandas as pd
import numpy as np
import argparse
from openai import OpenAI
import openai


# from incontext_examples import simple_incontext,lexicon_incontext,lexicon_noisy_incontext





def get_target_data(df,is_lexicon):
    df.fillna("",inplace=True)
    english = df["nl"].to_list()
    ltl = df["fl"].to_list()

    if(is_lexicon==1):
        # lexicon = df["pk_gold"].to_list()
        lexicon = df["parking"]
    
    if(is_lexicon==1):
        return {"english":english,"ltl":ltl,"lexicon":lexicon}
    else:
        return {"english":english,"ltl":ltl}


@backoff.on_exception(backoff.expo, openai.RateLimitError)
@backoff.on_exception(backoff.expo, openai.APIConnectionError)

def get_chatgpt_output(input,client,model):
    print("Start!")
    response = client.chat.completions.create(
      model=model,
      messages=[
          {"role":"user","content":input}
      ],
      temperature = 0
    )
    x = response.choices[0].message.content
    return x
    
    # if 'choices' in response:
    #     x = response['choices'][0].messsage.content.strip()
    #     if len(x) > 0:
    #         return response.choices[0].message.content.strip()
    #     else:
    #         return ''
    # else:
    #     return ''

def design_prompt(incontext,english,is_parking,lexicon=""):
    if(is_parking==1):
        #return f"{incontext}NL: {english}\nPK: {lexicon}\nREL_PK:"
        return f"{incontext}NL: {english}\nPK: {lexicon}\nLTL:"
    else:
        return incontext+"NL: "+english+"\nLTL:"

def incontext_examples(filename,is_parking):
    df = pd.read_csv(filename)
    df.fillna("",inplace=True)
    nl = df["nl"].to_list()
    fl = df["fl"].to_list()
    pk_retrieved = df["pk_retrieved"].to_list()
    pk_retrieved = [i.replace("<SEP>","|") for i in pk_retrieved]
    pk_gold = df["pk_gold"].to_list()
    number_of_incontext_examples = 3

    incontext_string = ""
    if(is_parking==0):
        for i in range(0,number_of_incontext_examples):
            incontext_string+=f"NL: {nl[i]}\nLTL: {fl[i]}\n"
    else:
        for i in range(0,number_of_incontext_examples):
            #incontext_string+=f"NL: {nl[i]}\nPK: {pk_retrieved[i]}\nREL: {pk_gold[i]}\nLTL: {fl[i]}\n"
            incontext_string+=f"NL: {nl[i]}\nPK: {pk_retrieved[i]}\nLTL: {fl[i]}\n"

    return incontext_string

if __name__ == "__main__":

    client = OpenAI(
        api_key= "",
        organization='',
    )

    parser = argparse.ArgumentParser(description='Model Arguments')
    parser.add_argument("-in","--input_file")
    parser.add_argument("-out","--output_file")
    parser.add_argument("-p",'--parking', nargs='?', const=0, type=int)
    parser.add_argument("-m","--model")

    args = parser.parse_args()

    df = pd.read_csv(args.input_file)

    incontext_string = incontext_examples("dataset/incontext_augmented.csv",args.parking)

    context = df["parking"].to_list()
    targets = get_target_data(df,args.parking)

    print(incontext_string)



    predictions = []

    for i in range(0,len(df)):
        if(args.parking==1):
            x = design_prompt(incontext_string,targets["english"][i],args.parking,targets["lexicon"][i])
        else:
            x = design_prompt(incontext_string,targets["english"][i],args.parking)
        
        # print(x)
        pred = get_chatgpt_output(x,client,args.model)

        predictions.append(pred)

        print(f"{i} datapoints done!")

        # print(x)
        # print("ANSWER!!!!")
        # print(pred)
        # break

    df = pd.DataFrame(data={"Generated Text":predictions,"Actual Text":targets["ltl"]})
    df.to_csv(args.output_file)