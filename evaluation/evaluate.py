from ..langhelper import open_file, json_to_file, create_folder_if_not_exists, print_eval
from math import sqrt
from re import compile
from collections import Counter

from os import getcwd


def text_to_vector(text):
    WORD = compile(r"\w+")
    words = WORD.findall(text)
    return Counter(words)



def get_cosine(vec1, vec2):
    vec1 = text_to_vector(vec1)
    vec2 = text_to_vector(vec2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = sqrt(sum1) * sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator



def jaccard_similarity(str1, str2):
    str1 = set(str1.split())
    str2 = set(str2.split())
    return float(len(str1 & str2)) / len(str1 | str2)






def get_value(percept, string):
    
    temp=string.split(percept)
    if len(temp) == 2:
        # print(temp)
        temp1=temp[1].strip().split()[0]
        try:
            num=int(float(temp1))
            return num
        except:
            # print("If")
            # print(f"Invalid value: {temp1}")
            # print(f"Invalid string: {string}\n\n{temp}\n\n{temp1}\n\n")
            return None
    else:
        list1=[x for x 	in temp if len(x)>0]
        # print(list1)
        for elem in list1:
            if elem[0] == " ":
                tmp=elem.strip().split()
                # print(len(tmp))
                if len(tmp)>0:
                    temp1=tmp[0]
                    try:
                        num=int(float(temp1))
                        return num
                    except:
                        # print("Else")
                        # print(f"Invalid value: {temp}")
                        # print(f"Invalid string: {string}\n\n{tmp}\n\n{temp}\n\n")
                        return None
                else:
                    return None
        return None





def get_position(string):
    temp=string.split("position")[1].strip().split()
    # print(temp)
    try:
        num1=int(temp[0])
        num2=int(temp[1])
        return num1, num2
    except:
        # print(f"num1:{num1} num2:{num2}\n\n{temp}\n\n")
        return None, None



def calc_domain_loss(p1, p2):
    p1=p1.replace("True", "1").replace("False", "0")
    p2=p2.replace("True", "1").replace("False", "0")

    percepts = ['is_demoed', 'on_ground', 'ball_touched', 'boost_amount', 'position', 'direction','speed', 'throttle', 'steer', 'jump', 'boost', 'handbrake']

    dom_bool = ['is_demoed', 'on_ground', 'ball_touched', 'throttle',
                    'steer', 'jump', 'boost', 'handbrake']

    dom1=['boost_amount']

    dom2=['direction']

    dom3=['speed']


    sum=0
    for p in percepts:
        if p in p1 and p in p2:
            if p in dom_bool:
                v1=get_value(p,p1)
                v2=get_value(p,p2)
                if v1 == None or v2 == None:
                    sum+=1
                elif v1 != v2:
                    sum+=1*0.5
            if p in dom1:
                v1=get_value(p,p1)
                v2=get_value(p,p2)
                if v1 == None or v2 == None:
                    sum+=1
                else:
                    sum+=abs(v1-v2)*0.01
            if p in dom2:
                v1=get_value(p,p1)
                v2=get_value(p,p2)
                if v1 == None or v2 == None:
                    sum+=1
                else:
                    sum+=abs(v1-v2)*0.002778
            if p in dom3:
                v1=get_value(p,p1)
                v2=get_value(p,p2)
                if v1 == None or v2 == None:
                    sum+=1
                else:               
                    sum+=abs(v1-v2)*0.0004348
            if p == 'position':
                x1,y1=get_position(p1)
                x2,y2=get_position(p2)
                if x1 != None and y1 != None:
                    sum+=abs(x1-y1)*0.000122
                else:
                    sum+=0.5
                if x2 != None and y2 != None:
                    sum+=abs(x2-y2)*0.0000976
                else:
                    sum+=0.5
        else:
            sum+=1
        # print(f"percept:{p}\nsum:{sum}\n")

    avg=sum/12

    return avg


def calc_loss(p1, p2):
    p1=p1.replace("True", "1").replace("False", "0")
    p2=p2.replace("True", "1").replace("False", "0")

    percepts = ['is_demoed', 'on_ground', 'ball_touched', 'boost_amount', 'position', 'direction','speed', 'throttle', 'steer', 'jump', 'boost', 'handbrake']

    bool_percepts = ['is_demoed', 'on_ground', 'ball_touched', 'throttle',
                    'steer', 'jump', 'boost', 'handbrake']

    ten_two_percepts=['boost_amount']

    ten_three_percepts=['direction']

    ten_four_percepts=['speed']


    sum=0
    for p in percepts:
        if p in p1 and p in p2:
            if p in bool_percepts:
                v1=get_value(p,p1)
                v2=get_value(p,p2)
                if v1 == None or v2 == None:
                    sum+=1
                elif v1 != v2:
                    sum+=1*0.1
            if p in ten_two_percepts:
                v1=get_value(p,p1)
                v2=get_value(p,p2)
                if v1 == None or v2 == None:
                    sum+=1
                else:   
                    sum+=abs(v1-v2)*0.01  
            if p in ten_three_percepts:
                v1=get_value(p,p1)
                v2=get_value(p,p2)
                if v1 == None or v2 == None:
                    sum+=1
                else:   
                    sum+=abs(v1-v2)*0.001
            if p in ten_four_percepts:
                v1=get_value(p,p1)
                v2=get_value(p,p2)
                if v1 == None or v2 == None:
                    sum+=1
                else:   
                    sum+=abs(v1-v2)*0.0001
            if p == 'position':
                x1,y1=get_position(p1)
                x2,y2=get_position(p2)
                if x1 != None and y1 != None:
                    sum+=abs(x1-y1)*0.00001
                else:
                    sum+=0.5
                if x2 != None and y2 != None:
                    sum+=abs(x2-y2)*0.00001
                else:
                    sum+=0.5
        else:
            sum+=1
        # print(f"percept:{p}\nsum:{sum}\n")

    avg=sum/12
    return avg




def combine_texts(target, nl, predicted):
    text = {
        'target': target,
        "nl": nl,
        'predicted': predicted,
        'cosine': get_cosine(target, predicted),
        'loss': calc_loss(target, predicted),
        'dloss': calc_domain_loss(target, predicted),
        'jaccard': jaccard_similarity(target, predicted)
    }    
    return text





def evaluate_results(ifile_path, ofile_path):
    print(f"file path {ifile_path}")
    output_folder=f"{getcwd()}/output/{ofile_path}".replace(".jsonl", "")
    results=open_file(ifile_path)

    print("file opened")
    create_folder_if_not_exists(output_folder)
    evaluated_results=[]
    count=0
    # print(results)
    # evaluated=''
    for result in results:
        evaluated=None
        result["nl"]=None
        result["predicted"]=None
        result["target"]=None
        count+=1
        target=result["target"]
        nl=result["nl"]
        predicted=result["predicted"]
        if result["target"] != None and result["nl"] != None and result["predicted"] != None:
            evaluated=combine_texts(target, nl, predicted)
        
        evaluated_results.append(evaluated)
        if count%10000==0:
            print(f"Evaluated Results: {count} out of {len(results)}\n")
            # print_eval(evaluated)
        
    
    json_to_file(evaluated_results, output_folder, ofile_path)

    return 0


