import sys
import os
from .ner.nertrainer import NERTrainer
from .translation.translationtrainer import TranslationTrainer
from .experiments import perform_experiment1, perform_experiment2
from .evaluation import evaluate_results




def get_help():
    help_str = "Include 5 arguments when calling for training:\n--task\n--model_checkpoint\n"
    help_str += "--dataset_name\n--model_name\n--num_epochs\n--mode\nThe arguments: --task, --model_checkpoint, "
    help_str += "--dataset_name, and --model_name are required.\n"
    help_str += "Example: python language-module --task=ner-nl --model_checkpoint=cw1521/model "
    help_str += "--dataset_name=cw1521/dataset --model_name=new-model"
    return help_str



def get_label_list():
    label_list = [
        "O",
        "LDEMO",
        "LB",
        "VBA",
        "LGROUND",
        "LBALL",
        "LSP",
        "VSPEED",
        "LD",
        "VDIR",
        "LBRAKE",
        "LSTE",
        "VSTEER",
        "LTHROT",
        "VTHROTTLE",
        "LBOOST",
        "LPOS"
    ]
    return label_list


def get_arg_dict_template():
    arg_dict = {}
    arg_dict["task"] = None
    arg_dict["mode"] = None
    arg_dict["model_checkpoint"] = None
    arg_dict["dataset_name"] = None
    arg_dict["model_name"] = None
    arg_dict["batch_size"] = None
    arg_dict["num_epochs"] = 10
    arg_dict["exp"]=None
    arg_dict["s1"]=None
    arg_dict["r1"]=None
    arg_dict["r2"]=None
    arg_dict["output"]=None
    return arg_dict


def get_arg_dict(args):
    arg_dict = get_arg_dict_template()
    for arg in args:
        arg_list = arg.split("=")
        var = arg_list[0].replace("--", "").lower()
        arg_dict[var] = arg_list[1]
    return arg_dict



def assert_valid_train_args(arg_dict):
    assert(arg_dict["task"] != None)
    assert(arg_dict["mode"] != None)
    assert(arg_dict["model_checkpoint"] != None)
    assert(arg_dict["dataset_name"] != None)
    assert(arg_dict["model_name"] != None)
    assert(arg_dict["num_epochs"] != None)




def is_arg_help(args):
    flag = False
    if len(args) == 1:
        arg = args[0]
        if arg == "--help" or arg == "-h":
            flag = True
    return flag



def train(arg_dict, token):
    task = arg_dict["task"]
    model_checkpoint = arg_dict["model_checkpoint"]
    dataset_name = arg_dict["dataset_name"]
    model_name = arg_dict["model_name"]
    test = True if arg_dict["mode"].lower() == "test" else False
    batch_size = arg_dict["batch_size"]

    if batch_size != None:
        try:
            batch_size = int(batch_size)
        except:
            raise TypeError

    try:
        num_epochs = int(arg_dict["num_epochs"])
    except:
        raise TypeError

    
    if task == "nl-ner":
        label_list = get_label_list()
        assert(label_list != None)            
        input = "tokens"
        target = "ner_ids"
        controller = NERTrainer(
            model_checkpoint,
            dataset_name,
            model_name,
            label_list,
            input,
            target,
            test,
            num_epochs,
            batch_size,
            token
        )
        controller.train()



    elif task == "ner-st":
        input = "ner_sentence"
        target = "state"
        controller = TranslationTrainer(
            model_checkpoint,
            dataset_name,
            model_name,
            input,
            target,
            test,
            num_epochs,
            batch_size,
            token
        )
        controller.train()

    elif task == "en-st":
        input = "sentence"
        target = "state"
        controller = TranslationTrainer(
            model_checkpoint,
            dataset_name,
            model_name,
            input,
            target,
            test,
            num_epochs,
            batch_size,
            token
        )
        controller.train()

    elif task == "st-en":
        input = "state"
        target = "sentence"
        controller = TranslationTrainer(
            model_checkpoint,
            dataset_name,
            model_name,
            input,
            target,
            test,
            num_epochs,
            batch_size,
            token
        )
        controller.train()
    
    else:
        print("Task currently unsupported.")




def main():
    hf_token = os.environ["HFAT"]
    args = sys.argv[1:]

    if is_arg_help(args):
        print(get_help())
    else:
        arg_dict = get_arg_dict(args)
        mode = arg_dict["mode"]
        
        if mode == "train" or mode == "test":
            assert_valid_train_args(arg_dict)
            train(arg_dict, hf_token)
        elif mode == "eval":
            ifile=arg_dict["ifile"]
            ofile=arg_dict["ofile"]
            evaluate_results(ifile, ofile)
        elif mode == "exp":
            exp=arg_dict["exp"]
            try:
                end=int(arg_dict["end"])
            except:
                end=-1
            if exp == "exp1":
                s1=arg_dict["s1"]
                r1=arg_dict["r1"]
                ds_name=arg_dict["dataset_name"]
                output_file=arg_dict["output"]
                perform_experiment1(s1, r1, ds_name, end, hf_token, output_file)
            elif exp == "exp2":
                s1=arg_dict["s1"]
                r1=arg_dict["r1"]
                r2=arg_dict["r2"]
                ds_name=arg_dict["dataset_name"]
                output_file=arg_dict["output"]
                perform_experiment2(s1, r1, r2, ds_name, end, hf_token, output_file)


    return 0





if __name__ == "__main__":
    main()