#!/bin/bash

curr_dir="$(pwd)"
script_dir="$(dirname "$(readlink -f "$0")")"
cd $script_dir/../..


# apptainer exec --nv ./hf-apptainer/hf.sif 
python3 -m language-module --task=nl-ner --model_checkpoint=distilbert-base-uncased --dataset_name=cw1521/nl-st --model_name=test-ner-1 --num_epochs=1 --mode=test --batch_size=32

cd $curr_dir