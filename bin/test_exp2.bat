SET script_path=%~dp0
SET curr_path=%CD%
cd %script_path%..\..

python -m language-module --mode=exp --exp=exp2 --s1=cw1521/st-en-10 --r1=cw1521/nl-ner-20 --r2=cw1521/ner-st-10 --dataset_name=cw1521/nl-st --output=nl-ner-st-10.jsonl

cd %curr_path%