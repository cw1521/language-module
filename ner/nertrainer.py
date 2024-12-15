from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments
from transformers import Trainer, DataCollatorForTokenClassification
from datasets import load_dataset
from evaluate import load
import numpy as np
import os


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
            batch_size,
            token
        ):
        self.token=token
        self.batch_size = batch_size
        self.model_checkpoint = model_checkpoint
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.input = input
        self.target = target
        self.label_list = label_list
        self.test = test   
        self.num_epochs = num_epochs
        self.dataset = self.get_dataset()           
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, token=self.token)
        id_label_map = self.get_id_label_map()
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint,
            token=self.token,
            num_labels=len(self.label_list),
            id2label=id_label_map["id2label"], 
            label2id=id_label_map["label2id"]
        )
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding='max_length',
            max_length=512
        )
        self.trainer = self.get_trainer()



    def train(self):
        self.trainer.train()
        self.trainer.save_model()
        self.trainer.save_state()


    def get_dataset(self):
        dataset = load_dataset(self.dataset_name)             
        if self.test:
            dataset["train"] = dataset["train"].shard(10, 0)
            dataset["validation"] = dataset["validation"].shard(10, 0)
            dataset["test"] = dataset["test"].shard(10, 0)
        elif self.dataset_name == "cw1521/nl-st-lg":
            dataset["train"] = dataset["train"].shard(50, 0)
            dataset["validation"] = dataset["validation"].shard(50, 0)
            dataset["test"] = dataset["test"].shard(50, 0)
        return dataset



    def get_tokenized_dataset(self):

        def align_labels_with_tokens(labels, word_ids):
            new_labels = []
            current_word = None
            for word_id in word_ids:
                if word_id != current_word:
                    # Start of a new word!
                    current_word = word_id
                    label = -100 if word_id is None else labels[word_id]
                    new_labels.append(label)
                elif word_id is None:
                    # Special token
                    new_labels.append(-100)
                else:
                    # Same word as previous token
                    label = labels[word_id]
            return new_labels
        
        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(
                examples["tokens"], truncation=True, is_split_into_words=True
            )
            all_labels = examples["ner_ids"]
            new_labels = []
            for i, labels in enumerate(all_labels):
                word_ids = tokenized_inputs.word_ids(i)
                new_labels.append(align_labels_with_tokens(labels, word_ids))

            tokenized_inputs["labels"] = new_labels
            return tokenized_inputs
        
        tokenized_dataset = self.dataset.map(tokenize_and_align_labels, batched=True, remove_columns=self.dataset["train"].column_names)
        train = tokenized_dataset["train"]
        valid = tokenized_dataset["validation"]

        return train, valid
    


    def compute_metrics(self, p):
        metric = load("seqeval")
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [[self.label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[self.label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

        results = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0.5)
        return {
            "precision": results["overall_precision"], 
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]
        }


    def get_id_label_map(self):
        id_label_map = {}
        id2label = {i: label for i, label in enumerate(self.label_list)}
        label2id = {v: k for k, v in id2label.items()}
        id_label_map["id2label"] = id2label
        id_label_map["label2id"] = label2id

        return id_label_map



    def get_training_args(self):
        if self.batch_size == None:
            batch_size = 8
        else:
            batch_size = self.batch_size
        args = TrainingArguments(
        f"./hf/models/{self.model_name}",
        save_steps=50,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=1e-7,
        logging_steps=50,
        num_train_epochs=self.num_epochs,
        logging_dir='./hf/logs',
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        save_total_limit=3
        )
        return args


    def get_trainer(self):
        train, valid = self.get_tokenized_dataset()
        args = self.get_training_args()
        train.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )
        valid.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train,
            eval_dataset=valid,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            data_collator=self.data_collator
        )
        return trainer

