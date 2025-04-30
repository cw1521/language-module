from transformers import AutoTokenizer, AutoModelForTokenClassification 
from transformers import AutoModelForSeq2SeqLM
from ..langhelper import print_log, create_folder_if_not_exists, print_result
from ..langhelper import write_log, get_dataset, json_to_file
from time import time
from os import getcwd


ner_id_map = {
    0: "O",
    1: "L-DEMO",
    2: "L-BA",
    3: "V-BA",
    4: "L-GROUND",
    5: "L-BALL",
    6: "L-SPEED",
    7: "V-SPEED",
    8: "L-DIR",
    9: "V-DIR",
    10: "L-BRAKE",
    11: "L-STEER",
    12: "V-STEER",
    13: "L-THROTTLE",
    14: "V-THROTTLE",
    15: "L-BOOST",
    16: "L-POS"
}


ner_tag_map = {
    "O": 0,
    "L-DEMO": 1,
    "L-BA": 2,
    "V-BA": 3,
    "L-GROUND": 4,
    "L-BALL": 5,
    "L-SPEED": 6,
    "V-SPEED": 7,
    "L-DIR": 8,
    "V-DIR": 9,
    "L-BRAKE": 10,
    "L-STEER": 11,
    "V-STEER": 12,
    "L-THROTTLE": 13,
    "V-THROTTLE": 14,
    "L-BOOST": 15,
    "L-POS": 16
}





def process_results(ds, sender_checkpoint, receiver1_checkpoint, receiver2_checkpoint, hf_token, output_folder, output_file):
    START_TIME=time()

    # Sender
    sender_model = AutoModelForSeq2SeqLM.from_pretrained(sender_checkpoint, token=hf_token)
    sender_tokenizer = AutoTokenizer.from_pretrained(sender_checkpoint, token=hf_token)

    # Receiver
    receiver1_model=AutoModelForTokenClassification.from_pretrained(receiver1_checkpoint, 
                                                                token=hf_token,
                                                                num_labels=len(ner_tag_map),
                                                                id2label=ner_id_map, 
                                                                label2id=ner_tag_map
                                                                )


    receiver1_tokenizer=AutoTokenizer.from_pretrained(receiver1_checkpoint, token=hf_token)

    receiver2_model=AutoModelForSeq2SeqLM.from_pretrained(receiver2_checkpoint, token=hf_token)
    receiver2_tokenizer=AutoTokenizer.from_pretrained(receiver2_checkpoint, token=hf_token)

    count=0
    ds_size=len(ds)
    results_list=[]
    # # Process dataset
    for data in ds:

        ###						Communication
        ############ Sender
        ## st-en
        st_en=data
  
        sender_encoded=sender_model.generate(**sender_tokenizer(st_en, return_tensors="pt", padding=True))
        sender_decoded=sender_tokenizer.decode(sender_encoded[0], skip_special_tokens=True)

   
        ####### Reciever

        ## NL-NER
        receiver1_encoded=receiver1_model(**receiver1_tokenizer(sender_decoded, return_tensors="pt", padding=True))
        
        logits=receiver1_encoded.logits
        predicted_token_class_ids = logits.argmax(-1)
        id_list=[x.item() for x in predicted_token_class_ids[0]]
        
        tokens=[ner_id_map[x] for x in id_list]
  
        ner_input=" ".join(tokens)


        # NER-St
        ner_st_encoded=receiver2_model.generate(**receiver2_tokenizer(ner_input,return_tensors="pt", padding=True))
        receiver2_decoded=receiver2_tokenizer.decode(ner_st_encoded[0], skip_special_tokens=True)
    

        result={}

        result["target"]=data
        result["nl"]=sender_decoded
        result["predicted"]=receiver2_decoded

        results_list.append(result)

        count+=1

        print_log(count, START_TIME, ds_size)
        if count % 100 == 0 or count == len(ds):
            write_log(count, ds_size, START_TIME, output_folder)
        if count % 500 == 0 or count == ds_size:
            json_to_file(results_list, output_folder, output_file)
            results_list=[]


    return results_list







def perform_experiment2(sender_checkpoint, receiver1_checkpoint, receiver2_checkpoint, dataset_name, sample_size, hf_token, output_file):
    # print(receiver1_checkpoint)
    output_folder=f"{getcwd()}/output/{output_file.replace('.jsonl', '')}"
    create_folder_if_not_exists(output_folder)

    ds=get_dataset(dataset_name, sample_size)

    output_file_name=f"{time()}_{output_file}"

    process_results(ds, sender_checkpoint, receiver1_checkpoint, receiver2_checkpoint, hf_token, output_folder, output_file_name)


    return 0