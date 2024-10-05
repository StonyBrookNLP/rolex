import numpy as np
import pandas as pd
import random
import requests
import tldr
from tqdm import tqdm
import subprocess

# # Replace 'ls' with your command
# command = ["ls", "-l"]  # Example command

# # Run the command and capture the output
# result = subprocess.run(command, stdout=subprocess.PIPE, text=True)

# # Get the output as a string
# output = result.stdout

# # Save the output to a file
# with open("output.txt", "w") as file:
#     file.write(output)

missing = {
    "acme":"acme\nMulti-platform cross assembler for 6502/6510/65816 CPU"
}



if __name__ == "__main__":
    # Open the documents csv
    docs = pd.read_csv("dataset/docs.csv")

    # Extract the list of unique doc ids
    doc_ids = docs["doc_id"].tolist()
    doc_contents = docs["doc_content"].tolist()

    #unique_cmds = {}
    cmds = {}
    cmd_list = []
    tldr_page_list = []

    #Extract the unique cmd names and all the docs for each cmd
    for i in range(0,len(doc_ids)):
        temp_id = doc_ids [i]
        divider_idx = temp_id.rfind("_")
        cmd_name,_ = temp_id[:divider_idx],temp_id[divider_idx+1:]
        
        if(".bck" in cmd_name):
            cmd_name = cmd_name.split(".")[0]
        
        if cmd_name not in cmds:
            cmds[cmd_name] = True

    for idx,cmd_name in tqdm(enumerate(cmds.keys())):
        command = f"tldr {cmd_name}"
        cmd_list.append(cmd_name)
        if(cmd_name in missing):
            tldr_page_list.append(missing[cmd_name])
        else:
            result = subprocess.run(command, stdout=subprocess.PIPE, text=True, shell=True)
            output = result.stdout
            tldr_page_list.append(output)
    
    df = pd.DataFrame({"cmd":cmd_list,"tldr_page":tldr_page_list})
    df.to_csv("dataset/tldr_pages.csv")
