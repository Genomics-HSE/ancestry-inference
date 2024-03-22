import json
import os
import pandas as pd
#from .genlink import get_classes, compute_simulation_params_fn
from os import listdir
from os.path import isfile, join

from .baseheuristic import getprobandmeanmatrices, composegraphs, checkpartition
from .ibdloader import load_pure


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
        print("means:", means)
        print("probs:", probs)
        
        
    return "dummy"

    
def collectandsaveparams(folder, outfile, override_popsizes):
    print(f"Collecting parameters for datasets from {folder}")
    if not(override_popsizes is None):
        print(f"Will save population sizes = {override_popsizes} instead of original")
    collected = collectparams(folder, override_popsizes)    
    with open(outfile,"w") as f:
        json.dump(collected,f)

    
def simulate(params):
    df = ""
    return df
    
    

def simulateandsave(paramfile, outfolder):
    print(f"Running simulations for parameters from {paramfile}")
    #create separate csv file for every record in paramfile 
    with open(paramfile,'r') as f:
        dct = json.load(f)
    for csvname in dct:
        simdf = simulate(dct[csvname])
        fname = os.path.join(outfolder, 'sim'+csvname)
        with open(fname, 'w') as f:
            #save dataframe here
            pass
        
    

if __name__=="__main__":
    print("just a test")