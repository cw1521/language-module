#!/bin/bash

# curr_dir="$(pwd)"
# script_dir="$(dirname "$(readlink -f "$0")")"
# cd $script_dir/../..
cd ../../

apptainer exec --nv ./hf-apptainer/hf.sif python3 -m language-module --task=ner-st --model_checkpoint=Helsinki-NLP/opus-mt-en-fr --dataset_name=cw1521/nl-st --model_name=ner-st-1 --num_epochs=1 --mode=train --batch_size=512


# cd $curr_dir