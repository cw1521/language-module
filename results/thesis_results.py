import json
import matplotlib.pyplot as plt



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
    epoch=""
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
        fname=f"data/{exp_type}-{i}.jsonl"
        ds=load_ds(fname)[:49999]
        stats=get_stats(ds)
        results[i]=stats
        avg_results[i]=stats["avg_results"]
    return results, avg_results


def display_line_graph(x, y, x_label, y_label, title):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()



def get_y_values(index, results, x_values):
    y=[]
    for x in x_values:
        y_val=results[x][index]
        print(f'{x}, {y_val}')
        y.append(y_val)
    return y



def main():
    st_nl_files=["10", "30", "40", "50"]
    st_ner_st=["20", "30"]
    st_nl_results, st_nl_avg=get_results(st_nl_files, "st-nl-st")
    st_nl_loss=get_y_values("loss", st_nl_avg, st_nl_files)
    display_line_graph(st_nl_files, st_nl_loss, "epoch", "loss",
                       "Loss from Experiment 1")


    # print(st_nl_results)






if __name__ == "__main__":
    main()
