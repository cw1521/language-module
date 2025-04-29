from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import sacremoses
from time import time
from os import getcwd
from ..langhelper import print_log, print_result, create_folder_if_not_exists
from ..langhelper import write_log, get_dataset, json_to_file




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

    

    for sender_out in sender_transcriber(ds):

    # for sender_out in sender_transcriber(ds):
        sender_message=sender_out[0]["translation_text"]
        # print(sender_message)
        receiver_message=receiver_transcriber(sender_message)[0]["translation_text"]

        result={}
        result["target"]=sender_message

        result["predicted"]=receiver_message

        results_list.append(result)

        count+=1

        if count % 500 == 0 or count == ds_size:
            json_to_file(results_list, output_folder, output_file)
            results_list=[]






def perform_experiment1(sender_checkpoint, receiver_checkpoint, dataset_name, sample_size, hf_token, output_file):
    output_folder=f"{getcwd()}/output/{output_file.replace('.jsonl', '')}"
    create_folder_if_not_exists(output_folder)

    # ds=get_dataset(dataset_name)[start:end]
    ds=get_dataset(dataset_name, sample_size)

    process_dataset(ds, sender_checkpoint, receiver_checkpoint, hf_token, output_folder, output_file)
    
    return 0

    

