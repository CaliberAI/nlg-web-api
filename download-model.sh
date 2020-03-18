#!/bin/bash
echo $MODEL_NAME
MODEL_NAME_TO_USE=${MODEL_NAME:-gpt2}
mkdir $MODEL_NAME_TO_USE
wget https://s3.amazonaws.com/models.huggingface.co/bert/$MODEL_NAME_TO_USE-config.json -O $MODEL_NAME_TO_USE/config.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/$MODEL_NAME_TO_USE-pytorch_model.bin -O $MODEL_NAME_TO_USE/pytorch_model.bin
wget https://s3.amazonaws.com/models.huggingface.co/bert/$MODEL_NAME_TO_USE-vocab.json -O $MODEL_NAME_TO_USE/vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/$MODEL_NAME_TO_USE-merges.txt -O $MODEL_NAME_TO_USE/merges.txt
