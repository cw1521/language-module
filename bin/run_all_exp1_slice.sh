#!/bin/bash


scripts=(
    "language-module/bin/run_exp_slice.sh 1 10 94499 -1"
    "language-module/bin/run_exp_slice.sh 1 20 93499 -1"
    "language-module/bin/run_exp_slice.sh 1 30 93599 -1"
    "language-module/bin/run_exp_slice.sh 1 40 93299 -1"
    "language-module/bin/run_exp_slice.sh 1 50 92899 -1"
)

for script in "${scripts[@]}"; do
    ./$script &
done 

disown
