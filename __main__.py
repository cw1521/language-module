import sys
from ner.nertrainer import NERTrainer
from translation.translationtrainer import TranslationTrainer
from os import path
from json import load



def get_help():
    help_str = "Include 4 arguments when calling programming:\n--task\n--model_checkpoint\n"
    help_str += "--dataset_name\n--model_name\n--num_epochs\n--mode\nThe arguments: --task, --model_checkpoint, "
    help_str += "--dataset_name, and --model_name are required.\n"
    help_str += "Example: python language-module --task=ner-nl --model_checkpoint=cw1521/model "
    help_str += "--dataset_name=cw1521/dataset --model_name=new-model"
    return help_str



def get_label_list():
    label_list = [
        "O",
        "L-DEMO",
        "L-BA",
        "V-BA",
        "L-GROUND",
        "L-BALL",
        "L-SPEED",
        "V-SPEED",
        "L-DIR",
        "V-DIR",
        "L-BRAKE",
        "L-STEER",
        "V-STEER",
        "L-THROTTLE",
        "V-THROTTLE",
        "L-BOOST",
        "L-POS"
    ]
    return label_list


def get_arg_dict_template():
    arg_dict = {}
    arg_dict["task"] = None
    arg_dict["mode"] = None
    arg_dict["model_checkpoint"] = None
    arg_dict["dataset_name"] = None
    arg_dict["model_name"] = None
    arg_dict["num_epochs"] = 10
    return arg_dict


def get_arg_dict(args):
    arg_dict = get_arg_dict_template()
    for arg in args:
        arg_list = arg.split("=")
        var = arg_list[0].replace("--", "").lower()
        arg_dict[var] = arg_list[1]
    return arg_dict



def assert_valid_args(arg_dict):
    assert(arg_dict["task"] != None)
    assert(arg_dict["mode"] != None)
    assert(arg_dict["model_checkpoint"] != None)
    assert(arg_dict["dataset_name"] != None)
    assert(arg_dict["model_name"] != None)





def is_arg_help(args):
    flag = False
    if len(args) == 1:
        arg = args[0]
        if arg == "--help" or arg == "-h":
            flag = True
    return flag



def train(arg_dict):
    task = arg_dict["task"]
    model_checkpoint = arg_dict["model_checkpoint"]
    dataset_name = arg_dict["dataset_name"]
    model_name = arg_dict["model_name"]
    test = arg_dict["mode"]
    

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
            num_epochs
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
            num_epochs
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
            num_epochs
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
            num_epochs
        )
        controller.train()
    
    else:
        print("Task currently unsupported.")





def main():
    home_path = path.dirname(path.abspath(sys.argv[0]))

    label_list_path = f"{home_path}\\language-module\\assets\\label_list.json"

    args = sys.argv[1:]

    if is_arg_help(args):
        print(get_help())
    else:
        arg_dict = get_arg_dict(args)
        arg_dict["label_list"] = get_json_from_file(label_list_path)["label_list"]
        assert_valid_args(arg_dict)
        mode = arg_dict["mode"]
        
        if mode == "train" or mode == "test-train":
            train(arg_dict)
        elif mode == "eval":
            return
        elif mode == "exp":
            return
    





if __name__ == "__main__":
    main()