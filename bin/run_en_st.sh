#!/bin/bash

curr_dir="$(pwd)"
script_dir="$(dirname "$(readlink -f "$0")")"
cd $script_dir/../..

python3 -m language-module --task=en-st --model_checkpoint=cw1521/en-st-lg-40 --dataset_name=cw1521/nl-st-lg --model_name=en-st-lg-50 --num_epochs=10 --mode=train

cd $script_dir