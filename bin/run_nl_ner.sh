cd ..
python3 -m language-module --task=nl-ner --model_checkpoint=bert-base-uncased --dataset_name=cw1521/nl-st --model_name=nl-ner-10 --num_epochs=10 --mode=train
