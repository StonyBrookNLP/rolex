## Run the following sequence of commands to download and prepare the dataset needed for training the models

```
python get_dataset.py
python get_tldr_pages.py
python augment_dataset.py
python create_retriever_json_data.py 
```


## The following commands trains the BGE retriever, generates the retriever evaluation file and then scores the retriever

```
export CUDA_VISIBLE_DEVICES=1 & 
nohup python -u -m torch.distributed.run --nproc_per_node 1 \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir models/bge_large \
--model_name_or_path BAAI/bge-large-en-v1.5 \
--train_data dataset/train_augmented.json \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size 32 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 256 \
--train_group_size 1 \
--negatives_cross_device \
--logging_steps 10 \
--query_instruction_for_retrieval "" > extra/bge_large.txt &

python -u parking_retriever_tldr.py -model models/bge_large -in dataset/val_augmented.csv -pk dataset/pk.csv -out dataset/test_bge_large.csv -k 10 -cuda 1

python score_retrieval.py
```

## The following command is an example of training the baseline model

```
python -u train_baseline_tldr.py -model Salesforce/codet5-small -dataset dataset/train_augmented.csv -val dataset/val_augmented.csv -output models/baseline_codet5_small -cuda 3 -bs 64 -lr 1e-4 -ep 10
```


## The following command creates the dataset for training and validating ROLex generators

```
python -u use_retriever_tldr.py -model models/bge_large -in dataset/train_augmented.csv -pk dataset/pk.csv -out dataset/train_augmented_retrieved.csv -k 10 -cuda 0 

python -u use_retriever_tldr.py -model models/bge_large -in dataset/val_augmented.csv -pk dataset/pk.csv -out dataset/val_augmented_retrieved.csv -k 10 -cuda 0 

python create_generator_dataset.py #for TRAINING
python create_generator_dataset.py #for VALIDATION
```
 
## The following commands (train_generator_XX) train the Code-T5 small generator models according to the different generator training schemes
```
#Basic
nohup python -u train_generator_basic.py -model Salesforce/codet5-small  -dataset dataset/train_generator.csv -val dataset/val_generator.csv -output models/parking_codet5_small -cuda 2 -bs 64 -lr 1e-4 -ep 5 > extra/t11.txt &

#Transfer Learning
python -u train_generator_weak_target.py -model Salesforce/codet5-small -dataset dataset/train_generator.csv -val dataset/val_generator.csv -output models/parking_codet5_small_weak -cuda 3 -bs 64 -lr 1e-4 -ep 10 

nohup python -u train_generator_basic.py -model models/parking_codet5_small_weak/model_files  -dataset dataset/train_generator.csv -val dataset/val_generator.csv -output models/parking_codet5_small -cuda 2 -bs 64 -lr 1e-4 -ep 10 > extra/t11.txt &
```

## The following commands help generate the test results for evaluating the models and placing them in the results folder


```
python -u parking_retriever_tldr.py -model models/bge_large -in dataset/test_augmented.csv -pk dataset/pk.csv -out dataset/test_parking.csv -k 10 -cuda 1 

#Example of testing the baseline model
nohup python -u evaluate_generate.py -file dataset/test_parking.csv  -model models/baseline_codet5_small/model_files  -pk 0 -cuda 2 -sv results/baseline_codet5_small.csv > extra/t10.txt &

#Example of testing the ROLex model
nohup python -u evaluate_generate.py -file dataset/test_parking.csv  -model models/parking_codet5_small/model_files  -pk 1 -cuda 2 -sv results/parking_codet5_small.csv
```

## Utilize the score_generation.py script afterwards to evaluate your results in the results folder