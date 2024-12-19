from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacremoses
from time import time
from os import getcwd
from ..langhelper import print_log, create_folder_if_not_exists, print_result
from ..langhelper import write_log, get_dataset, json_to_file



def process_dataset(ds, sender_checkpoint, receiver_checkpoint, hf_token, output_folder, output_file):
    START_TIME=time()

    sender_model = AutoModelForSeq2SeqLM.from_pretrained(sender_checkpoint, token=hf_token)
    sender_tokenizer = AutoTokenizer.from_pretrained(sender_checkpoint, token=hf_token)

    receiver_model = AutoModelForSeq2SeqLM.from_pretrained(receiver_checkpoint, token=hf_token)
    receiver_tokenizer = AutoTokenizer.from_pretrained(receiver_checkpoint, token=hf_token)


    count=0
    ds_size=len(ds)
    results_list=[]
    for text in ds:    
        # print(text)
        sender_translated=sender_model.generate(**sender_tokenizer(text, return_tensors="pt", padding=True))
        # print(sender_translated)
        sender_decoded=sender_tokenizer.decode(sender_translated[0], skip_special_tokens=True)
        # print(sender_decoded)
        receiver_translated=receiver_model.generate(**receiver_tokenizer(sender_decoded, return_tensors="pt", padding=True))
        receiver_decoded=receiver_tokenizer.decode(receiver_translated[0], skip_special_tokens=True)

        result={}
        result["target"]=text
        result["nl"]=sender_decoded
        result["predicted"]=receiver_decoded

        results_list.append(result)

        count+=1

        print_log(count, START_TIME, ds_size)

        if count % 2 == 0 or count == ds_size:
            write_log(count, len(ds), START_TIME, output_folder)
        if count % 5 == 0 or count == ds_size:
            json_to_file(results_list, output_folder, output_file)
            results_list=[]




def perform_experiment1(sender_checkpoint, receiver_checkpoint, dataset_name, hf_token, output_file):
    output_folder=f"{getcwd()}/{output_file.replace('.jsonl', '')}"
    create_folder_if_not_exists(output_folder)

    ds=get_dataset(dataset_name)

    process_dataset(ds, sender_checkpoint, receiver_checkpoint, hf_token, output_folder, output_file)
    
    return 0

    

