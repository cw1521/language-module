SET script_path=%~dp0
SET curr_path=%CD%
cd %script_path%..\..

python -m language-module --task=en-st --model_checkpoint=cw1521/opus-mt-en-st --dataset_name=cw1521/nl-st --model_name=test-trans-1 --num_epochs=1 --mode=test

cd %curr_path%