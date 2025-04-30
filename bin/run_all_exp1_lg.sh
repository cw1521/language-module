#!/bin/bash

cd $1

scripts=(
    "./language-module/bin/run_exp_lg.sh 1 10"
    "./language-module/bin/run_exp_lg.sh 1 20"
    "./language-module/bin/run_exp_lg.sh 1 30"
    "./language-module/bin/run_exp_lg.sh 1 40"
    "./language-module/bin/run_exp_lg.sh 1 50"
)

for script in "${scripts[@]}"; do
    bash "$script" & disown
done 

