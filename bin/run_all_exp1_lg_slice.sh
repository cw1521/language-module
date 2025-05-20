#!/bin/bash


scripts=(
    "language-module/bin/run_exp_lg_slice.sh 1 10 108999 -1"
    "language-module/bin/run_exp_lg_slice.sh 1 20 109299 -1"
    "language-module/bin/run_exp_lg_slice.sh 1 30 109299 -1"
    "language-module/bin/run_exp_lg_slice.sh 1 40 109299 -1"
    "language-module/bin/run_exp_lg_slice.sh 1 50 109299 -1"
)

for script in "${scripts[@]}"; do
    ./$script &
done 

disown
