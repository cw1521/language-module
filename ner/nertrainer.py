from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments
from transformers import Trainer, DataCollatorForTokenClassification
from datasets import load_dataset, load_metric
import numpy as np




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
            num_epochs
        ):
        
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
            tokenized_inputs = self.tokenizer(examples[self.input], truncation=True)
            tokenized_inputs["labels"] = examples[self.target]
            return tokenized_inputs

        tokenized_data = self.dataset.map(tokenize_and_align_labels, batched=True)
        train = tokenized_data["train"]
        valid = tokenized_data["validation"]
        return train, valid


    def get_training_args(self, num_epochs):
        batch_size = 256
        if self.test:
            batch_size = 64
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
        return {
            "precision": results["overall_precision"], 
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]
        }


    def get_trainer(self, num_epochs):
        train, valid = self.get_tokenized_datasets()
        args = self.get_training_args(num_epochs)

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

