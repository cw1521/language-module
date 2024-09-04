#!/bin/bash

# curr_dir="$(pwd)"
# script_dir="$(dirname "$(readlink -f "$0")")"
# cd $script_dir/../..

python3 language-module --task=st-en --model_checkpoint=cw1521/st-en-lg-38 --dataset_name=cw1521/nl-st-lg --model_name=st-en-lg-40 --num_epochs=2 --mode=train

# cd $script_dir