SET script_path=%~dp0
SET curr_path=%CD%
cd %script_path%..\..

python -m language-module --mode=exp --exp=exp1 --s1=cw1521/st-en-50 --r1=cw1521/en-st-50 --start=0 --end=1000 --dataset_name=cw1521/nl-st --output=st-nl-st-50.jsonl

cd %curr_path%