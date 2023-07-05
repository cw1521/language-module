
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments
import numpy as np
from transformers import Seq2SeqTrainer
from json import load
from os import getcwd




auth_token_path = f"{getcwd()}/auth_key.json"



def get_auth_key():
    with open(auth_token_path, "r") as f:
        key = load(f)
    return key["auth_key"]



auth_token = get_auth_key()


max_input_length = 128
max_target_length = 128

dataset_name = "cw1521/en-st"
model_name = "st-en-lg-40"
model_checkpoint = "cw1521/st-en-lg-30"





def get_dataset(name):
    train, valid = get_datafiles()
    return load_dataset(    
        name,
        data_files={'train':train, 'valid':valid},
        use_auth_token=auth_token,
        field="data"
    )




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






raw_data = get_dataset(dataset_name)

metric = load_metric("sacrebleu")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)



model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)



def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["input"], 
        max_length=max_input_length,
        truncation=True
    )
  
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target"],
            max_length=max_target_length,
            truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs






tokenized_data = raw_data.map(
    preprocess_function, batched=True, remove_columns=["input", "target"]
)


def get_training_args(num_epochs):
    train_batch_size = 64
    eval_batch_size = 64
    args = Seq2SeqTrainingArguments(
        model_name,
        save_steps=50,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        logging_dir='./logs',
    	gradient_accumulation_steps=4,
    	gradient_checkpointing=True,
	    tf32=True
    )
    return args





def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels



def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result



def get_trainer(num_epochs):
    args = get_training_args(num_epochs)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['valid'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    return trainer



trainer = get_trainer(10)

trainer.train()
model.save_pretrained(getcwd()+model_name)
