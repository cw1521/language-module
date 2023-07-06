import sys
from ner.nertrainer import NERTrainer
from translation.translationtrainer import TranslationTrainer
from os import path
from json import load




def get_help():
    help_str = "Include 4 arguments when calling programming:\n--task\n--model_checkpoint\n"
    help_str += "--dataset_name\n--model_name\n--num_epochs\nThe arguments: --task, --model_checkpoint, "
    help_str += "--dataset_name, and --model_name are required.\n"
    help_str += "Example: python language-module --task=ner-nl --model_checkpoint=cw1521/model "
    help_str += "--dataset_name=cw1521/dataset --model_name=new-model"
    return help_str



def get_json_from_file(path):
    with open(path, "r") as f:
        obj = load(f)
    return obj



def get_training_vars(args):
    task = None
    model_checkpoint = None
    dataset_name = None
    model_name = None
    test = False
    num_epochs = 10   

    for arg in args:
        arg_list = arg.split("=")
        # print(arg_list)
        if arg_list[0] == "--task":
            task = arg_list[1]
        elif arg_list[0] == "--model_checkpoint":
            model_checkpoint = arg_list[1]
        elif arg_list[0] == "--dataset_name":
            dataset_name = arg_list[1]
        elif arg_list[0] == "--model_name":
            model_name = arg_list[1]
        elif arg_list[0] == "--mode":
            if arg_list[1] == "test":
                test = True
        elif arg_list[0] == "--num_epochs":
            try:
                num_epochs = int(arg_list[1])
            except:
                print(f"Error: Cant convert {args[1]} to int.")

    assert(task != None)
    assert(model_checkpoint != None)
    assert(dataset_name != None)
    assert(model_name != None)
    return task, model_checkpoint, dataset_name, model_name, test, num_epochs





def is_arg_help(args):
    flag = False
    if len(args) == 1:
        arg = args[0]
        if arg == "--help" or arg == "-h":
            flag = True
    return flag






def main():
    home_path = path.dirname(path.abspath(sys.argv[0]))

    auth_token_path = f"{home_path}\\language-module\\assets\\auth_token.json"
    data_files_path = f"{home_path}\\language-module\\assets\\data_files.json"
    label_list_path = f"{home_path}\\language-module\\assets\\label_list.json"
    


    args = sys.argv[1:]

    if is_arg_help(args):
        print(get_help())
    else:
        training_vars = get_training_vars(args)

        task = training_vars[0]
        model_checkpoint = training_vars[1]
        dataset_name = training_vars[2]
        model_name = training_vars[3]
        test = training_vars[4]
        num_epochs =training_vars[5]
        auth_token = get_json_from_file(auth_token_path)["auth_token"]
        data_files = get_json_from_file(data_files_path)
        label_list = get_json_from_file(label_list_path)["label_list"]

        assert(data_files != None)

        if task == "nl-ner":
            assert(label_list != None)            
            input = "sentence"
            target = "ner_tags"
            controller = NERTrainer(
                model_checkpoint,
                dataset_name,
                model_name,
                auth_token,
                data_files,
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
                auth_token,
                data_files,
                input,
                target,
                test,
                num_epochs
            )
            controller.train()

        elif task == "en-st":
            input = "target"
            target = "input"
            controller = TranslationTrainer(
                model_checkpoint,
                dataset_name,
                model_name,
                auth_token,
                data_files,
                input,
                target,
                test,
                num_epochs
            )
            controller.train()

        elif task == "st-en":
            input = "input"
            target = "target"
            controller = TranslationTrainer(
                model_checkpoint,
                dataset_name,
                model_name,
                auth_token,
                data_files,
                input,
                target,
                test,
                num_epochs
            )
            controller.train()
        
        else:
            print("Task currently unsupported.")
    





if __name__ == "__main__":
    main()