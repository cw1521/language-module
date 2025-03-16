import json
import matplotlib.pyplot as plt
import pandas as pd



def load_ds(file_name):
    ds=[]
    with open(file_name) as f:
        for line in f:
            ds.append(json.loads(line))
    return ds


def get_avg(loss, dloss, jaccard, cosine, ds_size):
    avg_results={}
    avg_results["loss"]=loss/ds_size
    avg_results["dloss"]=dloss/ds_size
    avg_results["jaccard"]=jaccard/ds_size
    avg_results["cosine"]=cosine/ds_size
    return avg_results


def get_stats(ds):
    stats=init_results()
    for data in ds:
        stats["loss"]+=data["loss"]
        stats["dloss"]+=data["dloss"]
        stats["jaccard"]+=data["jaccard"]
        stats["cosine"]+=data["cosine"]
        stats["ds_size"]=len(ds)
        stats["avg_results"]=get_avg(stats["loss"], stats["dloss"],
                                stats["jaccard"],stats["cosine"], stats["ds_size"])
    return stats
    

def init_results():
    results={}
    results["loss"]=0    
    results["dloss"]=0
    results["jaccard"]=0
    results["cosine"]=0
    results["ds_size"]=0
    return results


def init_avg_results(fields):
    avg_result={}
    for field in fields:
        avg_result[field]=0
    return avg_result
    

def get_results(files, exp_type):
    results=init_results()
    avg_results=init_avg_results(files)
    for i in files:
        fname=f"./language-module/data/results-{exp_type}-{i}.jsonl"
        ds=load_ds(fname)
        stats=get_stats(ds)
        results[i]=stats
        avg_results[i]=stats["avg_results"]
    return avg_results


def display_line_graph(x, y, x_label, y_label, title):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()



def get_y_values(index, ds, x_values):
    y=[]
    for x in x_values:
        y_val=ds[x][index]
        y.append(y_val)
    return y

def show_table(df, title):
    pd.set_option("display.precision", 2)
    fig, ax=plt.subplots()
    table=pd.plotting.table(ax, df, loc="center")
    ax.axis("off")
    plt.tight_layout()
    plt.title(title, loc="center", y=0.7)
    plt.show()



def show_line_graph(field, ds, x_values, exp):
    y_values=get_y_values(field, ds, x_values)
    if field=="dloss":
        field="Domain Loss"
    display_line_graph(x_values, y_values, "epoch", field, 
                       f"{field} from Experiment {exp}")



def main():
    st_nl_files=["10", "30", "40", "50"]
    nl_ner_st_files=["10", "20", "30"]

    st_nl_avg=get_results(st_nl_files, "st-nl-st")
    nl_ner_avg=get_results(nl_ner_st_files, "nl-ner-st")

    st_nl_df=pd.DataFrame.from_dict(st_nl_avg).round(3)
    nl_ner_df=pd.DataFrame.from_dict(nl_ner_avg).round(3)


    show_table(st_nl_df, "Experiment 1")
    show_table(nl_ner_df, "Experiment 2")

    # Avg Loss
    show_line_graph("loss", st_nl_avg, st_nl_files, "1")

    # Avg Domain Loss
    show_line_graph("dloss", st_nl_avg, st_nl_files, "1")

    # Jaccard
    show_line_graph("jaccard", st_nl_avg, st_nl_files, "1")

    # Cosine
    show_line_graph("cosine", st_nl_avg, st_nl_files, "1")



    ### NL-NER-ST
    # Avg Loss
    show_line_graph("loss", nl_ner_avg, nl_ner_st_files, "2")

    # Avg Domain Loss
    show_line_graph("dloss", nl_ner_avg, nl_ner_st_files, "2")

    # Jaccard
    show_line_graph("jaccard", nl_ner_avg, nl_ner_st_files, "2")

    # Cosine
    show_line_graph("cosine", nl_ner_avg, nl_ner_st_files, "2")





