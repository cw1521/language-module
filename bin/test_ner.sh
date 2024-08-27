cd ..
python3 -m language-module --task=nl-ner --model_checkpoint=distilbert-base-uncased --dataset_name=cw1521/nl-st --model_name=test-ner-1 --num_epochs=1 --mode=test
