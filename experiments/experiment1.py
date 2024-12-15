from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import jsonlines
import sacremoses
import os
import time

# hf_token=os.environ["HFAT"]

# FILE_PATH="./output/st-nl-st-40-results.jsonl"



def get_dataset(dataset_name):
    data=load_dataset(dataset_name)
    ds=data["test"]["state"]
    return ds



def json_to_file(results_json, output_folder, file_name):
    file_path = f"{output_folder}/{file_name}"
    print("Writing output file...\n")
    with jsonlines.open(file_path, "a") as f:
        f.write_all(results_json)



def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)



def print_result(result):
    print(f"Target: {result['target']}\n")
    print(f"Predicted: {result['predicted']}\n")


def write_log(count, max_num, start_time, output_folder):
    create_folder_if_not_exists(f"{output_folder}/logs")
    print("Writing log...\n")
    with open(f"{output_folder}/logs/log-{start_time}.txt", "a") as f:
        f.write("State-Action")
        f.write(f"Results Completed: {count} out of {max_num}")
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        f.write(f"Elapsed time: {elapsed}")
        f.write(f"{round(count/max_num, 3)}% completed\n\n")








def process_dataset(ds, sender_checkpoint, receiver_checkpoint, hf_token, output_folder, output_file):
    START_TIME=time.time()

    sender_model = AutoModelForSeq2SeqLM.from_pretrained(sender_checkpoint, token=hf_token)
    sender_tokenizer = AutoTokenizer.from_pretrained(sender_checkpoint, token=hf_token)

    receiver_model = AutoModelForSeq2SeqLM.from_pretrained(receiver_checkpoint, token=hf_token)
    receiver_tokenizer = AutoTokenizer.from_pretrained(receiver_checkpoint, token=hf_token)


    count=0
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

        print("State-Action")
        print(f"Results Completed: {count} out of {len(ds)}")
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - START_TIME))
        print(f"Elapsed time: {elapsed}")
        print(f"{round(count/len(ds), 3)}% completed\n\n")


        if count % 100 == 0:
            write_log(count, len(ds), START_TIME, output_folder)
        if count % 500 == 0:
            json_to_file(results_list, output_folder, output_file)
            results_list=[]







def perform_experiment1(sender_checkpoint, receiver_checkpoint, dataset_name, hf_token, output_file):
    output_folder=f"{os.getcwd()}/{output_file.replace('.jsonl', '')}"
    create_folder_if_not_exists(output_folder)

    ds=get_dataset(dataset_name)

    process_dataset(ds, sender_checkpoint, receiver_checkpoint, hf_token, output_folder, output_file)

    return 0

    

