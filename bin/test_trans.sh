cd ..
python3 -m language-module --task=ner-st --model_checkpoint=cw1521/opus-mt-en-st --dataset_name=cw1521/nl-st --model_name=test-trans-1 --num_epochs=1 --mode=test
