import json
import os
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join


from .genlink import simulate_graph_fn, generate_matrices_fn
from .baseheuristic import getprobandmeanmatrices, composegraphs, checkpartition
from .ibdloader import load_pure

OFFSET = 8.0


def labeldict_to_labellist(labeldict):
    count = len(labeldict)
    result = [""] * count
    for lbl in labeldict:
        idx = labeldict[lbl]
        result[idx] = lbl
    return result

def collectparams(folder, override_popsizes):
    '''
    get every csv in the folder and compute its parameters
    
    dummy = {"CR.csv": {"pop_names":["pop1","pop2","pop3"],
                        "pop_sizes":[1000,1000,1000],
                        "edge_probability":[[1,0,0],[0,1,0],[0,0,1]],
                        "mean_weight":[[1,0,0],[0,1,0],[0,0,1]]
                       },
             "NC.csv": {"pop_names":["pop1","pop2","pop3"],
                        "pop_sizes":[1000,1000,1000],
                        "edge_probability":[[1,0,0],[0,1,0],[0,0,1]],
                        "mean_weight":[[1,0,0],[0,1,0],[0,0,1]]
                       }
            }
    '''
    
    filelist = [f for f in listdir(folder) if isfile(join(folder, f))]
    csvlist = [f for f in filelist if f[-3:] == 'csv']
    result = {}
    
    with open(join(folder, "meta.json"),"r") as f:
        meta = json.load(f)
    
    for fname in csvlist:        
        print("processing file", fname)
        fnamepath = join(folder, fname)
        pairs, weights, labels, labeldict, idxtranslator = load_pure( fnamepath, **meta[fname] )
        print("load ok!")
        graphdata = composegraphs(pairs, weights, labels, labeldict, idxtranslator)
        ncls = graphdata[0]['nodeclasses']
        grph = graphdata[0]['graph']
        trns = graphdata[0]['translation']

        checkpartition(grph, ncls, None, None, details=True, trns=trns)

        means, probs  = getprobandmeanmatrices(grph, ncls, labeldict)
        #print("means:", means)
        #print("probs:", probs)
        
        labellist = labeldict_to_labellist(labeldict)
        #print(labellist)
        #print(labels)
        pop_sizes = [ncls[lbl].shape[0] for lbl in labellist]
        probslist = []
        for elem in probs:
            probslist.append(list(elem))
        meanslist = []
        for elem in means:
            meanslist.append(list(elem))
        
        result[fname] = {
            "pop_names": labellist,
            "pop_sizes": pop_sizes,
            "edge_probability": probslist,
            "mean_weight":meanslist
        }
        
    print(result)    
    return result

    
def collectandsaveparams(folder, outfile, override_popsizes):
    print(f"Collecting parameters for datasets from {folder}")
    if not(override_popsizes is None):
        print(f"Will save population sizes = {override_popsizes} instead of original")
    collected = collectparams(folder, override_popsizes)    
    with open(outfile,"w") as f:
        json.dump(collected,f, indent=4, sort_keys=True)

        

def simulateandsave(paramfile, outfolder, seed):
    rng = np.random.default_rng(seed)
    print(f"Running simulations for parameters from {paramfile}")
    #create separate csv file for every record in paramfile 
    with open(paramfile,'r') as f:
        dct = json.load(f)
    for csvname in dct:
        fname = os.path.join(outfolder, 'sim'+csvname)
        params = dct[csvname]
        population_sizes = params["pop_sizes"]        
        offset = OFFSET
        edge_probs = np.array(params["edge_probability"])
        mean_weight = np.array(params["mean_weight"]) - offset #the next function wants corrected mean weights
        classes = params["pop_names"]
        counts, means, pop_index = generate_matrices_fn(population_sizes, offset, edge_probs, mean_weight, rng)                
        simulate_graph_fn(classes, means, counts, pop_index, fname)
        
    

if __name__=="__main__":
    print("just a test")