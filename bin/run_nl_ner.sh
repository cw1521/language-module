#!/bin/bash

# curr_dir="$(pwd)"
# script_dir="$(dirname "$(readlink -f "$0")")"
# cd $script_dir/../..

python3 -m language-module --task=nl-ner --model_checkpoint=bert-base-uncased --dataset_name=cw1521/nl-st --model_name=nl-ner-10 --num_epochs=10 --mode=train

# cd $curr_dir