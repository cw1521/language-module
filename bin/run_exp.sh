#!/bin/bash
python -m language-module --mode=exp --exp=exp$1 --s1=cw1521/st-en-$2 --r1=cw1521/en-st-$2 --dataset_name=cw1521/nl-st --sample_size=190000 --output=st-nl-st-$2.jsonl