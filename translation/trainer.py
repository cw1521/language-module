
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments
import numpy as np
from transformers import Seq2SeqTrainer
from json import load
from os import getcwd





class TranslationTrainer:
    auth_token_path = f"{getcwd()}\\language-module\\auth_key.json"

    def __init__(
            self,
            model_checkpoint,
            dataset_name,
            model_name,
            auth_token,
            data_files,
            num_epochs,
            input,
            target
        ):


        self.auth_token = auth_token
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.model_checkpoint = model_checkpoint
        self.data_files = data_files
        self.input = input
        self.target = target
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        self.trainer = self.get_trainer(num_epochs)



    def train(self):
        self.trainer.train()
        self.trainer.save_model()
        self.trainer.save_state()



    def get_dataset(self, name):
        train = self.data_files["train"]
        valid = self.data_files["valid"]
        return load_dataset(    
            name,
            data_files={"train":train, "valid":valid},
            use_auth_token=self.auth_token,
            field="data"
        )



    def get_tokenized_data(self):                
        def preprocess_function(examples):
            max_input_length = 128
            max_target_length = 128

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
        
        raw_data = self.get_dataset(self.dataset_name)        
        tokenized_data = raw_data.map(
            preprocess_function, batched=True, remove_columns=[self.input, self.target]
        )

        return tokenized_data



    def get_training_args(self, num_epochs):
        batch_size = 32
        args = Seq2SeqTrainingArguments(
            self.model_name,
            save_steps=50,
            evaluation_strategy = "epoch",
            learning_rate=1e-4,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=1e-5,
            save_total_limit=3,
            num_train_epochs=num_epochs,
            predict_with_generate=True,
            logging_dir='./logs',
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            tf32=True
        )
        return args



    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels




    def get_trainer(self, num_epochs):

        def compute_metrics(eval_preds):
            metric = load_metric("sacrebleu")
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            # Some simple post-processing
            decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
            result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            result = {"bleu": result["score"]}
            prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            return result

        args = self.get_training_args(num_epochs)
        data = self.get_tokenized_data()

        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=data['train'],
            eval_dataset=data['valid'],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )

        return trainer




