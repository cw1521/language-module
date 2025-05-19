#!/bin/bash


scripts=(
    "language-module/bin/run_eval1.sh 10"
    "language-module/bin/run_eval1.sh 20"
    "language-module/bin/run_eval1.sh 30"
    "language-module/bin/run_eval1.sh 40"
    "language-module/bin/run_eval1.sh 50"
    "language-module/bin/run_eval1_lg.sh 10"
    "language-module/bin/run_eval1_lg.sh 20"
    "language-module/bin/run_eval1_lg.sh 30"
    "language-module/bin/run_eval1_lg.sh 40"
    "language-module/bin/run_eval1_lg.sh 50"
)

for script in "${scripts[@]}"; do
    ./$script &
done 

disown

