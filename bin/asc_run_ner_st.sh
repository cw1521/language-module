#!/bin/bash

curr_dir="$(pwd)"
script_dir="$(dirname "$(readlink -f "$0")")"
cd $script_dir/../..



singularity exec --nv docker://huggingface/transformers-pytorch-gpu python3 -m language-module --task=ner-st --model_checkpoint=cw1521/opus-mt-en-st --dataset_name=cw1521/nl-st --model_name=ner-st-10 --num_epochs=10 --mode=train


cd $curr_dir