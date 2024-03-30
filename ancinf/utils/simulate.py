import json
import os
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join


from .genlink import simulate_graph_fn, generate_matrices_fn
from .baseheuristic import getprobandmeanmatrices, composegraphs, checkpartition, combinationgenerator
from .baseheuristic import getrandompermutation, dividetrainvaltest
from .runheuristic import translateconseqtodf
from .ibdloader import load_pure

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
    '''
    
    filelist = [f for f in listdir(folder) if isfile(join(folder, f))]
    csvlist = [f for f in filelist if f[-3:] == 'csv']
    result = {"experiment":{
                "valshare": 0.2,
                "testshare": 0.2,
                "partition_count": 10,
                "population_scale": [1.],
                "intra_edge_probability_scale": [1.],
                "extra_edge_probability_scale": [1.],
                "intra_weight_scale": [1.],
                "extra_weight_scale": [1.],
                "all_edge_probability_scale": [1.],
                "all_weight_scale": [1.]
               },
              "simulator":{
                  "type": "exponential",
                  "offset":8.0    
               },
              "datasets":{}}
    
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
        
        result["datasets"][fname] = {
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


def simulateandsaveonedataset(datasetparams, offset, fname, rng):        
    population_sizes = datasetparams["pop_sizes"]        
    edge_probs = np.array(datasetparams["edge_probability"])
    mean_weight = np.array(datasetparams["mean_weight"]) - offset #the next function wants corrected mean weights
    classes = datasetparams["pop_names"]
    counts, means, pop_index = generate_matrices_fn(population_sizes, offset, edge_probs, mean_weight, rng) 
    simulate_graph_fn(classes, means, counts, pop_index, fname)
        
def updateparams(datasetparams, experiment):
    resultparams = {}
    for k in datasetparams:
        resultparams[k] = datasetparams[k]
    
    new_mean_weight = np.array(datasetparams["mean_weight"])
    new_edge_probability = np.array(datasetparams["edge_probability"])
    
    new_mean_weight = new_mean_weight * experiment["all_weight_scale"]
    new_edge_probability = new_edge_probability * experiment["all_edge_probability_scale"]
    
    for idm in range(new_mean_weight.shape[0]):
        for idn in range(new_mean_weight.shape[0]):
            if idm == idn:
                new_mean_weight[idm, idn] *= experiment["intra_weight_scale"]
                new_edge_probability[idm, idn] *= experiment["intra_edge_probability_scale"]
            else:
                new_mean_weight[idm, idn] *= experiment["extra_weight_scale"]
                new_edge_probability[idm, idn] *= experiment["extra_edge_probability_scale"]
    #print(experiment["intra_edge_probability_scale"])
    #print(new_edge_probability)
    probslist = []
    for elem in new_edge_probability:
        probslist.append(list(elem))
    meanslist = []
    for elem in new_mean_weight:
        meanslist.append(list(elem))
    resultparams["mean_weight"] = meanslist
    resultparams["edge_probability"] = probslist
    
    new_pop_sizes = [ round(size * experiment["population_scale"]) for size in datasetparams["pop_sizes"] ]
    resultparams["pop_sizes"] = new_pop_sizes
    
    return resultparams
        

def savepartitions( datasetfname, valshare, testshare, partcount,  partfilename, rng):
    print("partitioning", datasetfname)
    partlist = []
    #load datafile 
    pairs, weights, labels, labeldict, idxtranslator = load_pure( datasetfname, debug=False )
    graphdata = composegraphs(pairs, weights, labels, labeldict, idxtranslator)
    ncls = graphdata[0]['nodeclasses']
    grph = graphdata[0]['graph']
    trns = graphdata[0]['translation']
    #translate indices in ncls to original indices
    
    
    
    for itr in range(partcount):        
        permt = getrandompermutation(ncls, rng)
        trainnodeclasses, valnodeclasses, testnodeclasses = dividetrainvaltest(ncls, valshare, testshare, permt)        
        part_ok, part_errors = checkpartition(grph, trainnodeclasses, valnodeclasses, testnodeclasses, details=False, trns=trns)
        if not part_ok:
            print("bad partition on iter", itr)
        traindf, valdf, testdf = translateconseqtodf(trns, trainnodeclasses, valnodeclasses, testnodeclasses)
        traindflist = {}
        valdflist = {}
        testdflist = {}
        for c in traindf:
            traindflist[c] = traindf[c].tolist()
        for c in valdf:
            valdflist[c] = valdf[c].tolist()
        for c in testdf:
            testdflist[c] = testdf[c].tolist()
        
        partlist.append({"train": traindflist, "val": valdflist, "test": testdflist})
    
    #print(partlist)
    with open(partfilename,"w") as f:
        json.dump({"partitions":partlist}, f, indent=4, sort_keys=True)
    
    
def simulateandsave(paramfile, outfolder, seed):
    rng = np.random.default_rng(seed)
    print(f"Running simulations for parameters from {paramfile}")
    with open(paramfile,'r') as f:
        dct = json.load(f)  
    experiments = dct["experiment"]
    datasets = dct["datasets"]
    simparams = dct["simulator"]
    offset = simparams["offset"]
    
    expfiledict = {}
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    #iterate through datasets
    for datasetname in datasets:
        experimentlist = []
        #now iterate through experiments
        expnum = 0 
        
        expgen = combinationgenerator(experiments)
        datasetparams = datasets[datasetname]
            
        for experiment in expgen:
            datafilename = os.path.join(outfolder, 'sim'+datasetname+str(expnum)+'.csv' )
            partfilename = os.path.join(outfolder, 'sim'+datasetname+str(expnum)+'.part' )
            
            updateddatasetparams = updateparams(datasetparams, experiment)
            simulateandsaveonedataset(updateddatasetparams, offset, datafilename, rng)
            
            savepartitions(datafilename, experiment["valshare"], experiment["testshare"], experiment["partition_count"], partfilename, rng)
            
            expdict = { "id":expnum,
                       "experiment":experiment,
                       "datasetparams":datasetparams,
                       "datafile":datafilename,
                       "partitionfile":partfilename
            }
            experimentlist.append(expdict)
            expnum += 1
        expfiledict[datasetname] = experimentlist 
        
    experimentlistfile = os.path.join(outfolder, 'experimentlist.json' )
    #create file with all the experiments descritption
    with open(experimentlistfile,"w") as f:
        json.dump(expfiledict, f, indent=4, sort_keys=True)
        
if __name__=="__main__":
    print("just a test")