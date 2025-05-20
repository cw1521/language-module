#!/bin/bash

python -m language-module --mode=exp --exp=exp$1 --s1=cw1521/st-en-lg-$2 --r1=cw1521/en-st-lg-$2 --dataset_name=cw1521/nl-st-lg --start==$3 --end==$4 --output=st-nl-st-lg-$2.jsonl