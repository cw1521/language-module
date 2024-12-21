SET script_path=%~dp0
SET curr_path=%CD%
cd %script_path%..\..

python -m language-module --mode=eval --ifile=language-module\assets\st-nl-st-20-results.jsonl --ofile=st-nl-st-20-eval.jsonl
cd %curr_path%