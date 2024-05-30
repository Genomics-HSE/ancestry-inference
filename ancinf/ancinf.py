import click
import numpy as np
import time
from multiprocessing import Pool

from .utils import simulate as sim 
from .utils import runheuristic
import json, os

#@click.option("--count", default=1, help="Number of greetings.")
#@click.option("--name", prompt="Your name", help="The person to greet.")
#click.echo(f"Hello, {folder}, {out}!")



@click.group()
def cli():
    pass


#STAGE1 GETPARAMS
@cli.command()
@click.argument("datadir")
@click.argument("workdir")
@click.option("--infile", default="project.ancinf", help="Project file, defaults to project.ancinf")
@click.option("--outfile", default=None, help="Output file with simulation parameters, defaults to project file with '.params' extension")
def getparams(datadir, workdir, infile, outfile):
    """Collect parameters of csv files in the DATADIR listed in project file from WORKDIR"""    
    if outfile is None:
        #try to remove .ancinf from infile
        position = infile.find('.ancinf')
        if position>0:
            outfile = infile[:position]+'.params'
        else:
            outfile = infile+'.params'
    sim.collectandsaveparams(datadir, workdir, infile, outfile)
    print("Finished!")

    
#STAGE1' PREPROCESS    
@cli.command()
@click.argument("datadir")
@click.argument("workdir")
@click.option("--infile", default="project.ancinf", help="Project file, defaults to project.ancinf")
@click.option("--outfile", default=None, help="Output file with experiment list, defaults to project file with '.explist' extension")
@click.option("--seed", default=2023, help="Random seed")
def preprocess(datadir, workdir, infile, outfile, seed):
    """Filter datsets from DATADIR, generate train-val-test splits and experiment list file in WORKDIR"""    
    if outfile is None:
        #try to remove .ancinf from infile
        position = infile.find('.ancinf')
        if position>0:
            outfile = infile[:position]+'.explist'
        else:
            outfile = infile+'.explist'
    rng = np.random.default_rng(seed)
    start = time.time()
    sim.preprocess(datadir, workdir, infile, outfile, rng)
    print(f"Finished! Total {time.time()-start:.2f}s")


#STAGE 2 SIMULATE
@cli.command()
@click.argument("workdir")
@click.option("--infile", default="project.params", help="File with simulation parameters, defaults to project.params")
@click.option("--outfile", default=None, help="Output file with experiment list, defaults to project file with '.explist' extension")
@click.option("--seed", default=2023, help="Random seed")
def simulate(workdir, infile, outfile, seed):
    """Generate ibd graphs, corresponding slpits and experiment list file for parameters in INFILE"""    
    if outfile is None:
        #try to remove .ancinf from infile
        position = infile.find('.params')
        if position>0:
            outfile = infile[:position]+'.explist'
        else:
            outfile = infile+'.explist'
            
    rng = np.random.default_rng(seed)
    start = time.time()
    sim.simulateandsave(workdir, infile, outfile, rng)
    print(f"Finished! Total {time.time()-start:.2f}s")
    
    
# #STAGE3 HEURISTICS GNN etc
def combine_splits(partresults):    
    result = {}
    for partres in partresults:
        #include values from partresults
        for dataset in partres:            
            if dataset in result:
                #existing dataset. list experiments and find new
                existing_exp_ids = [ exp["exp_idx"] for exp in result[dataset] ]
                for exp in partres[dataset]: 
                    if exp["exp_idx"] in existing_exp_ids:
                        #existing experiment, find it
                        for res_exp in result[dataset]:
                            if res_exp["exp_idx"] == exp["exp_idx"]:
                                break
                        res_exp["dataset_time"] += exp["dataset_time"]
                        #now list classifiers and add split scores
                        #classifiers should be the same
                        for classifier in exp["classifiers"]:
                            for metric in exp["classifiers"][classifier]:
                                if metric!="class_scores":
                                    res_exp["classifiers"][classifier][metric]["values"].extend(exp["classifiers"][classifier][metric]["values"])        
                                else:
                                    for pop in exp["classifiers"][classifier]["class_scores"]:
                                        res_exp["classifiers"][classifier][metric][pop]["values"].extend(exp["classifiers"][classifier][metric][pop]["values"])        
                            
                    else:
                        #new experiment
                        result[dataset].append(exp)
                        result[dataset][-1]["dataset_begin"] = "multiprocessing"
                        result[dataset][-1]["dataset_end"] = "multiprocessing"
                        
            else:
                #new dataset
                result[dataset] = partres[dataset]
                for exp in result[dataset]:
                    exp["dataset_begin"] = "multiprocessing"
                    exp["dataset_end"] = "multiprocessing"
            #print(result[dataset])
    #recompute mean and std
    for dataset in result:
        for exp in result[dataset]:
            for classifier in exp["classifiers"]:
                for metric in exp["classifiers"][classifier]:
                    metricresults = exp["classifiers"][classifier][metric]
                    if metric!="class_scores":
                        metricresults["mean"] = np.average(metricresults["values"])
                        metricresults["std"] = np.std(metricresults["values"])
                        if "clean_mean" in metricresults:
                            metricresults["clean_mean"] = np.average(metricresults["clean_values"])
                            metricresults["clean_std"] = np.std(metricresults["clean_values"])
                    else: 
                        for cl in metricresults:
                            metricresults[cl]["mean"] = np.average(metricresults[cl]["values"])
                            metricresults[cl]["std"] = np.std(metricresults[cl]["values"])
                        
                        
    return {"brief": sim.getbrief(result), "details":result}

def runandsavewrapper(args):
    return sim.runandsaveall(args["workdir"], args["infile"], args["outfilebase"], args["fromexp"], args["toexp"], 
                             args["fromsplit"], args["tosplit"], args["gpu"])
    
#STAGE5 TEST HEURISTICS, COMMUNITY DETECTIONS AND TRAIN&TEST NNs
@cli.command()
@click.argument("workdir")
@click.option("--infile", default="project.explist", help="File with experiment list, defaults to project.explist")
@click.option("--outfile", default=None, help="File with classification metrics, defaults to project file with '.result' extension")
@click.option("--seed", default=2023, help="Random seed")
@click.option("--processes", default=1, help="Number of parallel workers")
@click.option("--fromexp", default=None, help="The first experiment to run")
@click.option("--toexp", default=None, help="Last experiment (not included)")
@click.option("--fromsplit", default=None, help="The first split to run")
@click.option("--tosplit", default=None, help="Last split (not included)")
@click.option("--gpu", default=0, help="GPU")
@click.option("--gpucount", default=1, help="GPU count")
def crossval(workdir, infile, outfile, seed, processes, fromexp, toexp, fromsplit, tosplit, gpu, gpucount):
    """Run crossvalidation for classifiers including heuristics, community detections, GNNs and MLP networks"""     
    rng = np.random.default_rng(seed)  
    if outfile is None:
        #try to remove .ancinf from infile
        position = infile.find('.explist')
        if position>0:
            outfilebase = infile[:position]
        else:
            outfilebase = infile
    else:
        outfilebase = outfile
    
    start = time.time()            
    if processes == 1:
        sim.runandsaveall(workdir, infile, outfilebase, fromexp, toexp, fromsplit, tosplit, gpu)
    else:              
        #get every process only one job computing splitrange aforehead
        splitcount = int(tosplit)-int(fromsplit)
        splitsperproc = splitcount // processes
        splitincrements = [splitsperproc]*processes
        for idx in range(splitcount % processes):
            splitincrements[idx] +=1
        splitrange = [int(fromsplit)] 
        incr = 0
        for idx in range(processes):
            incr += splitincrements[idx]
            splitrange.append(int(fromsplit)+incr)
            
        
        print("Split seprarators:", splitrange)
        
        taskargs = [{"workdir":workdir, 
                     "infile":infile, 
                     "outfilebase":outfilebase, 
                     "fromexp":fromexp, 
                     "toexp":toexp, 
                     "fromsplit":splitrange[procnum], 
                     "tosplit":splitrange[procnum+1], 
                     "gpu":procnum%gpucount}  for procnum in range(processes)]
        print(taskargs)
        
        with Pool(processes) as p:
            resfiles = p.map(runandsavewrapper, taskargs)
        
        #now combine results        
        if (fromexp is None) and (toexp is None):
            outfile_exp_postfix = ""
        else:
            outfile_exp_postfix = "_e" + str(fromexp) +"-"+ str(toexp)  
        outfile_split_postfix = "_s" + str(fromsplit) +"-"+ str(tosplit)
        outfilename = outfilebase+outfile_exp_postfix+outfile_split_postfix+'.results'
        partresults = []
        for partresultfile in resfiles:
            with open(partresultfile,"r") as f:
                partresults.append(json.load(f)["details"])
        combined_results=combine_splits(partresults)        
        
        with open(os.path.join(workdir, outfilename),"w", encoding="utf-8") as f:
            json.dump(combined_results, f, indent=4, sort_keys=True)
        
    print(f"Finished! Total {time.time()-start:.2f}s.")    


#STAGE 4 INFERENCE
@cli.command()
@click.argument("workdir")
@click.argument("traindf")
@click.argument("inferdf")
@click.argument("model")
@click.argument("weights")
def infer(workdir, traindf, inferdf, model, weights):
    """
    traindf: Dataset on which the model was trained
    
    inferdf, Dataset with nodes with classes to be inferred (labelled 'unknown')
    
    model: Model name
    
    weights: Weights file
    """
    nodes, labels = sim.inference(workdir, traindf, inferdf, model, weights)
    outfilename = inferdf+".inferred"
    result = {"node_"+str(node):lbl for node, lbl in zip(nodes, labels)}
        
    with open(os.path.join(workdir, outfilename),"w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, sort_keys=True)


def main():
    cli()