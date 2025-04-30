from datasets import load_dataset
from jsonlines import open as jlopen
from time import time, strftime, gmtime
from os import path, makedirs
import random


def get_dataset(dataset_name, sample_size):
    ds=load_dataset(dataset_name)
    ds=ds["test"]["state"]
    used_index=[]
    new_ds=[]
    ds_size=len(ds)-1
    while len(new_ds) < sample_size:
        index=random.randint(0, ds_size)
        if index not in used_index:
            used_index.append(index)
            new_ds.append(ds[index])
    # print(new_ds)
    return new_ds


# def data(ds):
#     for data in ds:
#         yield data



def json_to_file(results_json, output_folder, file_name):
    file_path = f"{output_folder}/{file_name}"
    print(f"Writing output file {file_path}...\n")
    with jlopen(file_path, "a") as f:
        f.write_all(results_json)



def create_folder_if_not_exists(folder_path):
    if not path.exists(folder_path):
        makedirs(folder_path)

def print_eval(eval):
    print(f"Target: {eval['target']}\n")
    print(f"NL: {eval['nl']}\n")
    print(f"Predicted: {eval['predicted']}\n")
    print(f"SAVL: {eval['loss']}\n")
    print(f"DELF: {eval['dloss']}\n")
    print(f"Cosine: {eval['cosine']}\n")
    print(f"Jaccard: {eval['jaccard']}\n\n")


def print_result(result):
    print(f"Target: {result['target']}\n")
    print(f"Predicted: {result['predicted']}\n")


def write_log(count, max_num, start_time, output_folder):
    create_folder_if_not_exists(f"{output_folder}/logs")
    print("Writing log...\n")
    with open(f"{output_folder}/logs/log-{start_time}.txt", "a") as f:
        f.write("State-Action\n")
        f.write(f"Results Completed: {count} out of {max_num}\n")
        elapsed = strftime("%H:%M:%S", gmtime(time() - start_time))
        f.write(f"Elapsed time: {elapsed}\n")
        f.write(f"{round(count/max_num, 3)}% completed\n\n")


def print_log(count, start_time, max_num):
    print("State-Action")
    print(f"Results Completed: {count} out of {max_num}")
    elapsed = strftime("%H:%M:%S", gmtime(time() - start_time))
    print(f"Elapsed time: {elapsed}")
    print(f"{round(count/max_num, 3)}% completed\n\n")



def open_file(file_path):
    file=[]
    with jlopen(file_path) as f:
        for line in f:
            file.append(line)
    return file