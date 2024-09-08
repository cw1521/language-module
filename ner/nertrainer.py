from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments
from transformers import Trainer, DataCollatorForTokenClassification
from datasets import load_dataset, load_metric
# from evaluate import load
import numpy as np
import subprocess
import sys
# from seqeval.metrics import accuracy_score
# from seqeval.metrics import classification_report
# from seqeval.metrics import f1_score

class NERTrainer:
    test = False

    def __init__(
            self,
            model_checkpoint,
            dataset_name,
            model_name,
            label_list,
            input,
            target,
            test,
            num_epochs,
            batch_size
        ):
        self.batch_size = batch_size
        self.model_checkpoint = model_checkpoint
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.input = input
        self.target = target
        self.label_list = label_list 
        self.test = test   
        self.dataset = self.get_dataset()           
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint,
            num_labels=len(self.label_list)
        )
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding='max_length',
            max_length=512
        )
        self.trainer = self.get_trainer(num_epochs)



    def train(self):
        self.trainer.train()
        self.trainer.save_model()
        self.trainer.save_state()


    def get_dataset(self):
        dataset = load_dataset(self.dataset_name)             
        if self.test:
            dataset["train"] = dataset["train"].shard(10, 0)
            dataset["valid"] = dataset["validation"].shard(10, 0)
        return dataset


    def get_tokenized_datasets(self):

        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(examples[self.input], truncation=True, is_split_into_words=True)
            labels = []
            for i, label in enumerate(examples[self.target]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(label[word_idx])
                    previous_word_idx = word_idx
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        tokenized_data = self.dataset.map(tokenize_and_align_labels, batched=True)
        train = tokenized_data["train"]
        valid = tokenized_data["validation"]
        return train, valid


    def compute_metrics(self, p):
        metric = load_metric("seqeval")
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [[self.label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[self.label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"], 
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]
        }




    def get_training_args(self, num_epochs):
        # if self.test:
        if self.batch_size == None:
            batch_size = 8
        else:
            batch_size = self.batch_size
        args = TrainingArguments(
        f"./hf/models/{self.model_name}",
        save_steps=50,
        eval_strategy = "epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=1e-5,
        save_total_limit=3,
        num_train_epochs=num_epochs,
        logging_dir='./hf/logs',
        gradient_accumulation_steps=4
        # fp16=True
        )
        # else:
        #     if self.batch_size == None:
        #         batch_size = 32
        #     else:
        #         batch_size = self.batch_size
        #     args = TrainingArguments(
        #         f"models/{self.model_name}",
        #         save_steps=50,
        #         evaluation_strategy = "epoch",
        #         learning_rate=1e-4,
        #         per_device_train_batch_size=batch_size,
        #         per_device_eval_batch_size=batch_size,
        #         weight_decay=1e-5,
        #         save_total_limit=3,
        #         num_train_epochs=num_epochs,
        #         logging_dir='./logs',
        #         gradient_accumulation_steps=4,
        #         fp16=True
        #)
        return args


    def get_trainer(self, num_epochs):
        train, valid = self.get_tokenized_datasets()
        args = self.get_training_args(num_epochs)
        train.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
                )
        valid.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

        trainer = Trainer(
            self.model,
            args,
            train_dataset=train,
            eval_dataset=valid,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            data_collator=self.data_collator
        )
        return trainer

