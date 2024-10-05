## Commands to generate the synthetic data and the set of lexicon for training

```
python generate.py
python create_lexicon.py
python create_retriever_training_data.py

python create_retriever_json_data.py 
```

## The following commands trains the BGE retriever, generates the retriever evaluation file and then scores the retriever

```
CUDA_VISIBLE_DEVICES=3 \ 
nohup python -u -m torch.distributed.run --nproc_per_node 1 \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir models/bge_large \
--model_name_or_path BAAI/bge-large-en-v1.5 \
--train_data dataset/retriever_train.json \
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
--query_instruction_for_retrieval "" 

python -u use_retriever_synthetic.py -model models/bge_large -in dataset/test_augmented.csv -pk dataset/test_pk.csv -out dataset/test_bge_large.csv -k 10 -cuda 0 
```

## The following command is an example of training the baseline model

```
python -u train_baseline_synthetic.py -model Salesforce/codet5-small -dataset dataset/baseline_train.csv -val dataset/baseline_train.csv -output models/baseline_codet5_small -cuda 3 -bs 64 -lr 1e-4 -ep 5 
```


## The following command creates the dataset for training and validating ROLex generators
```
python create_lexicon.py

python -u use_retriever_synthetic.py -model models/bge_large -in dataset/RAG_train.csv -pk dataset/train_pk.csv -out dataset/train_augmented_retrieved.csv -k 10 -cuda 2 

python -u use_retriever_synthetic.py -model models/bge_large -in dataset/val_augmented.csv -pk dataset/test_pk.csv -out dataset/val_augmented_retrieved.csv -k 10 -cuda 2 

python create_generator_dataset.py
```

## The following commands is an example (train_generator_XX) of training the Code-T5 small generator models according to the different generator training schemes

```
#Basic
python -u train_generator_basic.py -model Salesforce/codet5-small  -dataset dataset/train_generator.csv -val dataset/train_generator.csv -output models_2/parking_codet5_small -cuda 2 -bs 64 -lr 1e-4 -ep 5

#Transfer Learning
python -u train_generator_weak_target.py -model Salesforce/codet5-small -dataset dataset/train_generator.csv -val dataset/train_generator.csv -output models/parking_codet5_small_weak -cuda 2 -bs 64 -lr 1e-4 -ep 5 

python -u train_generator_basic.py -model models/parking_codet5_small_weak/model_files  -dataset dataset/train_generator.csv -val dataset/train_generator.csv -output models/parking_codet5_small -cuda 2 -bs 64 -lr 1e-4 -ep 5


```


## The following commands help generate the test results for evaluating the models and placing them in the results folder

```
python -u parking_retriever_synthetic.py -model models/bge_large -in dataset/nfs_rfc_COLM.csv -pk dataset/nfs_rfc_pk.csv -out dataset/test_nfs_rfc.csv -k 10 -cuda 2


python -u evaluate_generate.py -file dataset/test_nfs_rfc.csv  -model models/baseline_codet5_small/model_files  -pk 0 -cuda 2 -sv results_nfs_rfc/baseline_codet5_small.csv 



python -u evaluate_generate.py -file dataset/test_nfs_rfc.csv  -model models/parking_codet5_small/model_files  -pk 1 -cuda 2 -sv results_nfs_rfc/parking_codet5_small.csv 
```


## The following commands generate the results for the incontext learning scenario

```
python -u use_retriever_synthetic.py -model models/bge_large -in dataset/incontext_examples.csv -pk dataset/incontext_pk.csv -out dataset/incontext_augmented.csv -k 10 -cuda 2

python -u incontext_eval.py -in dataset/test_nfs_rfc.csv -out results_incontext/chatgpt_parking.csv -p 1 -m gpt-3.5-turbo> t1.txt &
```

## Utilize the score_generation_xx.py script afterwards to evaluate your results in the results folder