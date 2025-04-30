from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from datasets import load_dataset
import sacremoses
from time import time, strftime, gmtime
from os import getcwd
from ..langhelper import print_log, print_result, create_folder_if_not_exists
from ..langhelper import write_log, data, json_to_file




def process_dataset(ds, sender_checkpoint, receiver_checkpoint, hf_token, output_folder, output_file):
    START_TIME=time()

    sender_model = AutoModelForSeq2SeqLM.from_pretrained(sender_checkpoint, token=hf_token)
    sender_tokenizer = AutoTokenizer.from_pretrained(sender_checkpoint, token=hf_token)

    receiver_model = AutoModelForSeq2SeqLM.from_pretrained(receiver_checkpoint, token=hf_token)
    receiver_tokenizer = AutoTokenizer.from_pretrained(receiver_checkpoint, token=hf_token)

    sender_transcriber = pipeline(task="translation", model=sender_model, tokenizer=sender_tokenizer)
    receiver_transcriber = pipeline(task="translation", model=receiver_model, tokenizer=receiver_tokenizer)

  

    results_list=[]

    count=0
    ds_size=len(ds)
    


    for sender_out in sender_transcriber(data(ds)):

        sender_message=sender_out[0]["translation_text"]

        receiver_message=receiver_transcriber(sender_message)[0]["translation_text"]

        result={}
        result["target"]=ds[count]

        result["predicted"]=receiver_message

        results_list.append(result)

        count+=1

        if count % 100 == 0 or count == ds_size:
            json_to_file(results_list, output_folder, output_file)
            results_list=[]






def perform_experiment1(sender_checkpoint, receiver_checkpoint, dataset_name, sample_size, hf_token, output_file):
    output_folder=f"{getcwd()}/output/{output_file.replace('.jsonl', '')}"
    create_folder_if_not_exists(output_folder)

    ds=load_dataset(dataset_name, split="test")["state"]

    output_file_name=f"{strftime('%H:%M:%S', gmtime(time()))}_{output_file}"

    process_dataset(ds, sender_checkpoint, receiver_checkpoint, hf_token, output_folder, output_file_name)
    
    return 0

    

