SET script_path=%~dp0
SET curr_path=%CD%
cd %script_path%..\..

python -m language-module --task=nl-ner --model_checkpoint=distilbert-base-uncased --dataset_name=cw1521/nl-st --model_name=test-ner-1 --num_epochs=1 --mode=test

cd %curr_path%