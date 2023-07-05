#########################################################
# Load Modules
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments
from transformers import DataCollatorForTokenClassification, Trainer
from datasets import load_dataset, load_metric
import numpy
from json import load
from os import getcwd
import pandas as pd
import numpy as np


def get_auth_key(path):
    with open(path, "r") as f:
        key = load(f)
    return key["auth_key"]





##########################################################
# Constants

auth_token_path = f"{getcwd()}\\auth_key.json"

model_checkpoint = "distilbert-base-uncased"
dataset_name = "cw1521/en-st-ner-small"
model_name = "nl-ner-sm-10"



auth_token = get_auth_key(auth_token_path)
output_path = f"{getcwd()}\\output\\{model_name}"




#########################################################
# Conversion Maps
ner_id_map = {
    "0": "O",
    "1": "L-DEMO",
    "2": "L-BA",
    "3": "V-BA",
    "4": "L-GROUND",
    "5": "L-BALL",
    "6": "L-SPEED",
    "7": "V-SPEED",
    "8": "L-DIR",
    "9": "V-DIR",
    "10": "L-BRAKE",
    "11": "L-STEER",
    "12": "V-STEER",
    "13": "L-THROTTLE",
    "14": "V-THROTTLE",
    "15": "L-BOOST",
    "16": "L-POS"
  }

ner_tag_map = {
    "O": 0,
    "L-DEMO": 1,
    "L-BA": 2,
    "V-BA": 3,
    "L-GROUND": 4,
    "L-BALL": 5,
    "L-SPEED": 6,
    "V-SPEED": 7,
    "L-DIR": 8,
    "V-DIR": 9,
    "L-BRAKE": 10,
    "L-STEER": 11,
    "V-STEER": 12,
    "L-THROTTLE": 13,
    "V-THROTTLE": 14,
    "L-BOOST": 15,
    "L-POS": 16
  }

label_list =  [
    "O",
    "L-DEMO",
    "L-BA",
    "V-BA",
    "L-GROUND",
    "L-BALL",
    "L-SPEED",
    "V-SPEED",
    "L-DIR",
    "V-DIR",
    "L-BRAKE",
    "L-STEER",
    "V-STEER",
    "L-THROTTLE",
    "V-THROTTLE",
    "L-BOOST",
    "L-POS"
]







#########################################################
# Load Dataset
def get_datafiles():
    train = [
    'oracle-train1.json',
    'oracle-train2.json',
    'oracle-train3.json',
    'oracle-train4.json',
    'oracle-train5.json',
    'oracle-train6.json',
    'oracle-train7.json',
    'oracle-train8.json',
    'oracle-train9.json',
    'oracle-train10.json'
    ]   

    valid = ['oracle-valid.json']
    return train, valid

def get_dataset(name):
    train, valid = get_datafiles()
    return load_dataset(    
        name,
        data_files={'train':train, 'valid':valid},
        use_auth_token=auth_token,
        field="data"
    )

dataset = get_dataset(dataset_name)






#########################################################
# Load Tokenizer, Model, and Data Collator

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(ner_id_map))
data_collator = DataCollatorForTokenClassification(tokenizer)






#########################################################
# Tokenize Datasets

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["sentence"], truncation=True)
    tokenized_inputs["labels"] = examples["ner_tags"]
    return tokenized_inputs


train_tokenized_datasets = dataset["train"].map(tokenize_and_align_labels, batched=True)
valid_tokenized_datasets = dataset["valid"].map(tokenize_and_align_labels, batched=True)




#########################################################
# Metrics
metric = load_metric("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}
    




#########################################################
# Trainer Arguments and Trainer
def get_training_args(num_epochs):
    batch_size = 64
    args = TrainingArguments(
        model_name,
        save_steps=50,
        evaluation_strategy = "epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=1e-5,
        save_total_limit=3,
        num_train_epochs=num_epochs,
        logging_dir='./logs',
    	gradient_accumulation_steps=4,
	    tf32=True
    )
    return args


args = get_training_args(10)

trainer = Trainer(
    model,
    args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=valid_tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)






#########################################################
# Train, Evaluate, Save Model
trainer.train()
trainer.evaluate()
trainer.save_model()


