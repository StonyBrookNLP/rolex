# Importing libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
# Importing the T5 modules from huggingface/transformers
#from transformers import T5Tokenizer, T5ForConditionalGeneration, RobertaTokenizer,AutoTokenizer,AutoModelForCausalLM,AutoModelForSeq2SeqLM
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForSeq2SeqLM
# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console
#For command line argument parsing
import sys
import getopt
import argparse
from torch import cuda


# to display dataframe in ASCII format
def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)

class NL2CODEDataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]
        
        
    def __len__(self):
        """returns the length of dataframe"""
        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",return_tensors="pt",
            add_special_tokens=True
        )

        
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


def train(epoch, tokenizer, model, device, loader, optimizer):

    """
    Function to be called for training with the parameters passed from main function

    """

    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            labels=y,
        )
        loss = outputs.loss

        if _ % 100 == 0:
            training_logger.add_row(str(epoch), str(_), str(loss))
            console.print(training_logger)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(epoch, tokenizer, model, device, loader):

  """
  Function to evaluate model for predictions

  """
  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask,
              max_length=512,
              num_beams=5,
              repetition_penalty=2.5,
              length_penalty=1.0,
              early_stopping=True
              )
          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
          
          if _%50==0:
              console.print(f'Completed {_}')

          predictions.extend(preds)
          actuals.extend(target)
  return predictions, actuals


def T5Trainer(
    dataframe, val_df, source_text, target_text, model_params, output_dir):

    """
    T5 trainer

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = AutoTokenizer.from_pretrained(model_params["MODEL"])
    tokenizer.add_tokens("NL2CODE")
    tokenizer.add_tokens("NL")
    tokenizer.add_tokens("<SEP>")
    tokenizer.add_tokens("PK")
    
    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = AutoModelForSeq2SeqLM.from_pretrained(model_params["MODEL"])
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]
    val_df = val_df[[source_text, target_text]]
    display_df(dataframe.head(2))
    display_df(val_df.head(2))

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    # train_size = 0.8
    train_dataset = dataframe[0:40000]
    val_dataset = val_df[50000:50200].reset_index(drop=True)

    console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TEST Dataset: {val_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = NL2CODEDataSetClass(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    val_set = NL2CODEDataSetClass(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

     # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    # Training loop
    console.log(f"[Initiating Fine Tuning]...\n")

    for epoch in range(model_params["EPOCHS"]):
        #Train for one epoch
        train(epoch, tokenizer, model, device, training_loader, optimizer)

        #Save each model epoch
        path = os.path.join(output_dir, "model_files_"+str(epoch))
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

        #Save the predictions
        print(f"Performing validation for epoch {epoch}")
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv(os.path.join(output_dir, f"predictions_{epoch}.csv"))

        print(f"{epoch} is complete!")

    console.save_text(os.path.join(output_dir, "logs.txt"))

    console.print(f"""Training Complete\n""")


def arguments():
    parser = argparse.ArgumentParser(description='Model Arguments')
    parser.add_argument("-model","--model")
    parser.add_argument("-dataset","--dataset")
    parser.add_argument("-val","--val_dataset")
    parser.add_argument("-output","--output_path")
    parser.add_argument('-cuda','--cuda_device')
    parser.add_argument('-bs','--batch_size',type=int)
    parser.add_argument('-lr','--learning_rate',type=float)
    parser.add_argument('-ep','--epochs',type=int)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = arguments()

    print(args)

    #Open the file and process the dataset
    df = pd.read_csv(args.dataset)
    print(len(df))
    df.fillna("",inplace=True)

    #Open the validation file and process the dataset
    val_df = pd.read_csv(args.val_dataset)
    print(len(val_df))
    val_df.fillna("",inplace=True)

    #Create the input to the model
    df["case_1"] = "NL2CODE\nNL:"+df["nl"]+"\nPK:"+df["weak_pk"]
    df["case_2"] = "NL2CODE\nNL:"+df["nl"]+"\nPK:"+df["pk_generator"]

    case_1 = df["case_1"].tolist()
    case_2 = df["case_2"].tolist()
    out = df["fl"].tolist()

    new_input = []
    target = []

    for i in range(0,len(case_1)):

        new_input.append(case_1[i])
        target.append(out[i])

        new_input.append(case_2[i])
        target.append(out[i])

    new_df = pd.DataFrame({"input":new_input,"target":target})



    #Create the validation input to the model
    val_df["case_1"] = "NL2CODE\nNL:"+val_df["nl"]+"\nPK:"+val_df["pk_generator"]

    case_1 = val_df["case_1"].tolist()
    out = val_df["fl"].tolist()

    new_input = []
    target = []

    for i in range(0,len(case_1)):
        new_input.append(case_1[i])
        target.append(out[i])

    new_val_df = pd.DataFrame({"input":new_input,"target":target})

    # Setting up the device for GPU usage
    if(args.cuda_device!=-1):
        device_name = 'cuda:'+args.cuda_device
    else:
        device_name = 'cuda'
    device = device_name if cuda.is_available() else 'cpu'

    # define a rich console logger and training logger to log training progress
    console = Console(record=True)
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII
    )

    #let's define model parameters specific to T5
    model_params = {
        "MODEL": args.model,  # model_type: t5-base/t5-large
        "TRAIN_BATCH_SIZE": args.batch_size,  # training batch size
        "VALID_BATCH_SIZE": args.batch_size,  # validation batch size
        "EPOCHS": args.epochs,  # number of training epochs
        "LEARNING_RATE": args.learning_rate,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 256,  # max length of target text
        "SEED": 42,  # set seed for reproducibility
    }

    T5Trainer(
        dataframe=new_df,
        val_df = new_val_df,
        source_text="input",
        target_text="target",
        model_params=model_params,
        output_dir=args.output_path
    )