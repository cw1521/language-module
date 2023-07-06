from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments
from transformers import DataCollatorForTokenClassification, Trainer
from datasets import load_dataset, load_metric
from json import load
from os import getcwd
import numpy as np



class NlNer:
    auth_token_path = f"{getcwd()}\\language-module\\auth_key.json"

    def __init__(self, model_checkpoint, dataset_name, model_name, num_epochs):
        self.model_checkpoint = model_checkpoint
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.auth_token = self.get_auth_key(self.auth_token_path)
        self.output_path = f"{getcwd()}\\output\\{self.model_name}"
        self.dataset = self.get_dataset(self.dataset_name)
        self.label_list = self.get_label_list()               
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint,
            num_labels=len(self.get_ner_id_map())
        )
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)

        self.trainer = self.get_trainer(num_epochs)



    def train(self):
        self.trainer.train()
        self.trainer.save_model(f"output\\{self.model_name}")
        self.trainer.save_state()



    def get_auth_key(self, path):
        with open(path, "r") as f:
            key = load(f)
        return key["auth_key"]



    def get_ner_id_map(self):
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
        return ner_id_map
    



    def get_ner_tag_map(self):
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
        return ner_tag_map




    def get_label_list(self):
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
        return label_list




    def get_datafiles(self):
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



    def get_dataset(self, name):
        train, valid = self.get_datafiles()
        return load_dataset(    
            name,
            data_files={'train':train, 'valid':valid},
            use_auth_token=self.auth_token,
            field="data"
        )




    def get_training_args(self, num_epochs):
        batch_size = 128
        args = TrainingArguments(
            self.model_name,
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

    def compute_metrics(self, p):
        metric = load_metric("seqeval")
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [[self.label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[self.label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}



    def get_trainer(self, num_epochs):
        train, valid = self.get_tokenized_datasets()
        args = self.get_training_args(num_epochs)
        trainer = Trainer(
            self.model,
            args,
            train_dataset=train,
            eval_dataset=valid,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        return trainer




    def get_tokenized_datasets(self):
        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(examples["sentence"], truncation=True)
            tokenized_inputs["labels"] = examples["ner_tags"]
            return tokenized_inputs
        train = self.dataset["train"].map(tokenize_and_align_labels, batched=True)
        valid = self.dataset["valid"].map(tokenize_and_align_labels, batched=True)

        return train, valid
















