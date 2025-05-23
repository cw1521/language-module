#!/bin/bash

module purge
module load anaconda/3-2024.02

source activate /home/usacxw001/.env

cd /home/usacxw001

scripts=(
    "python -m language-module --mode=exp --exp=exp1 --s1=cw1521/st-en-10 --r1=cw1521/en-st-10 --dataset_name=cw1521/nl-st --start=94499 --end==-1 --output=st-nl-st-10.jsonl"

    "python -m language-module --mode=exp --exp=exp1 --s1=cw1521/st-en-20 --r1=cw1521/en-st-20 --dataset_name=cw1521/nl-st --start=93499 --end==-1 --output=st-nl-st-20.jsonl"

    "python -m language-module --mode=exp --exp=exp1 --s1=cw1521/st-en-30 --r1=cw1521/en-st-30 --dataset_name=cw1521/nl-st --start=93599 --end==-1 --output=st-nl-st-30.jsonl"

    "python -m language-module --mode=exp --exp=exp1 --s1=cw1521/st-en-40 --r1=cw1521/en-st-40 --dataset_name=cw1521/nl-st --start=93299 --end==-1 --output=st-nl-st-40.jsonl"

    "python -m language-module --mode=exp --exp=exp1 --s1=cw1521/st-en-50 --r1=cw1521/en-st-50 --dataset_name=cw1521/nl-st --start=92899 --end==-1 --output=st-nl-st-50.jsonl"
)



for script in "${scripts[@]}"; do
    $script &
done 

