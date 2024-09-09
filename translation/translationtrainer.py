
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_dataset
from evaluate import load
import numpy as np



class TranslationTrainer:
    test = False

    def __init__(
            self,
            model_checkpoint,
            dataset_name,
            model_name,
            input,
            target,
            test,
            num_epochs,
            batch_size
        ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.model_checkpoint = model_checkpoint
        self.input = input
        self.target = target
        self.test = test  
        self.dataset = self.get_dataset()  
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
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
        return dataset


    def get_tokenized_dataset(self):                
        def preprocess_function(examples):
            max_input_length = 512
            max_target_length = 512

            model_inputs = self.tokenizer(
                examples[self.input],
                max_length=max_input_length,
                truncation=True
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    examples[self.target],
                    max_length=max_target_length,
                    truncation=True
                )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
     
        tokenized_data = self.dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=[self.input, self.target]
        )
        train = tokenized_data["train"]
        valid = tokenized_data["validation"]
        return train, valid
    
        
    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels


    def compute_metrics(self, eval_preds):
        metric = load("sacrebleu")
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


    def get_training_args(self):
            if self.batch_size == None:
                batch_size = 32
            else:
                batch_size = self.batch_size
            args = Seq2SeqTrainingArguments(
            f"./hf/models/{self.model_name}",
            save_steps=50,
            evaluation_strategy = "epoch",
            learning_rate=1e-4,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=1e-5,
            save_total_limit=3,
            num_train_epochs=self.num_epochs,
            predict_with_generate=True,
            logging_dir='./hf/logs',
            gradient_accumulation_steps=4,
            gradient_checkpointing=True
            )
            return args


    def get_trainer(self):
        args = self.get_training_args()
        train, valid = self.get_tokenized_dataset()
        train.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )
        valid.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )
        
        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=train,
            eval_dataset=valid,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            data_collator=self.data_collator
        )
        return trainer
