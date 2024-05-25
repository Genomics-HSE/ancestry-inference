from sklearn.metrics import f1_score

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import os
import json

from . import ibdloader
from . import baseheuristic as bh



def translatedftoconseq(trns, dftrainnodes, dfvalnodes, dftestnodes):
    '''
        gets train val and test nodes in dataset enumeration and returns
        consecutive enumeration
    '''
    trainnodeclasses, valnodeclasses, testnodeclasses = {}, {}, {}
    sorter = np.argsort(trns)
    for c in dftrainnodes:        
        trainnodeclasses[c] = sorter[np.searchsorted(trns, dftrainnodes[c], sorter=sorter)] 
        valnodeclasses[c] = sorter[np.searchsorted(trns, dfvalnodes[c], sorter=sorter)] 
        testnodeclasses[c] = sorter[np.searchsorted(trns, dftestnodes[c], sorter=sorter)] 
    
    return trainnodeclasses, valnodeclasses, testnodeclasses



def translateconseqtodf(trns, trainnodeclasses, valnodeclasses, testnodeclasses):
    '''
        gets train val and test nodes in consequtive enumeration and returns
        datafile enumeration
    '''
    dftrainnodes, dfvalnodes, dftestnodes = {}, {}, {}
    for c in trainnodeclasses:
        dftrainnodes[c] = trns[trainnodeclasses[c]]    
        dfvalnodes[c] = trns[valnodeclasses[c]]
        dftestnodes[c] = trns[testnodeclasses[c]]
    
    return dftrainnodes, dfvalnodes, dftestnodes 

def getf1macro(grph, labels, labeldict, pairs, trns, trainnodeclasses, valnodeclasses, testnodeclasses):
    trainvalnodeclasses = {}
    for c in trainnodeclasses:
        trainvalnodeclasses[c] = np.concatenate((trainnodeclasses[c], valnodeclasses[c]))
    trainnodes, valnodes, testnodes = bh.gettrainvaltestnodes(trainnodeclasses, valnodeclasses, testnodeclasses)    
    testlabels = labels[testnodes]
    featuredict = bh.getfeatures(grph, testnodes, trainvalnodeclasses, labeldict, pairs, trns )
    simplepredictions = bh.getsimplepred(featuredict)
    
    result = {}
    for feature in simplepredictions: 
        prediction = simplepredictions[feature][testnodes]        
        result[feature] =  f1_score(testlabels, prediction, average='macro')
    return result

#internal randomize
def collectmacrosforrandompartitions(grph, labels, labeldict, pairs, trns, ncls, rng, itercount, valshare, testshare ):
    #1. IBD sum average for different partitioning
    # that is, testing features on 20% (testshare) nodes only, and features are based on the links to the rest 80% (train+val) of nodes only    
    featuremacro = {
        'IbdSumPerEdge': [], 
        'IbdSum': [], 
        'LongestIbd': [], 
        'SegmentCount': [], 
        'EdgeCountPerClassize': [], 
        'EdgeCount': []}

    for itr in range(itercount):        
        permt = bh.getrandompermutation(ncls, rng)
        trainnodeclasses, valnodeclasses, testnodeclasses = bh.dividetrainvaltest(ncls, valshare, testshare, permt)        
        part_ok, part_errors = bh.checkpartition(grph, trainnodeclasses, valnodeclasses, testnodeclasses, details=False, trns=trns)
        if not part_ok:
            print("bad partition on iter", itr)
            #for msg in part_errors:
            #    print(msg)
        
        f1s = getf1macro(grph, labels, labeldict, pairs, trns, trainnodeclasses, valnodeclasses, testnodeclasses)
        for feature in f1s:
            featuremacro[feature].append(f1s[feature])   
    return featuremacro
    
#or external saved partitioning
def collectmacrosforstoredpartitions(grph, labels, labeldict, pairs, trns, ncls, rng, partitions, conseq):
    featuremacro = {
        'IbdSumPerEdge': [], 
        'IbdSum': [], 
        'LongestIbd': [], 
        'SegmentCount': [], 
        'EdgeCountPerClassize': [], 
        'EdgeCount': []}
    
    for idx, partition in enumerate(partitions):
        if conseq:
            trainnodeclasses = partition['train']
            valnodeclasses = partition['val']
            testnodeclasses = partition['test']
        else:
            dftrainnodes = partition['train']
            dfvalnodes = partition['val']
            dftestnodes = partition['test']
            trainnodeclasses, valnodeclasses, testnodeclasses = translatedftoconseq(trns, dftrainnodes, dfvalnodes, dftestnodes)
        
        part_ok, part_errors = bh.checkpartition(grph, trainnodeclasses, valnodeclasses, testnodeclasses, details=False, trns=trns)
        if not part_ok:
            print("bad partition on iter", idx)
            for msg in part_errors:
                print(msg)
        
        f1s = getf1macro(grph, labels, labeldict, pairs, trns, trainnodeclasses, valnodeclasses, testnodeclasses)
        for feature in f1s:
            featuremacro[feature].append(f1s[feature])   
    return featuremacro

def run(datafile, valshare=None, testshare=None, itercount=None, partitions=None, conseq=False, debug=True, filter_params = None, rng=None):
    if debug:
        print("====================================")
    
    if filter_params is None:
        pairs, weights, labels, labeldict, idxtranslator = ibdloader.load_pure( datafile, debug=debug )
    else: 
        pairs, weights, labels, labeldict, idxtranslator = ibdloader.load_pure( datafile, debug=debug, **(filter_params) )
    
    graphdata = bh.composegraphs(pairs, weights, labels, labeldict, idxtranslator)
    ncls = graphdata[0]['nodeclasses']
    grph = graphdata[0]['graph']
    trns = graphdata[0]['translation']        

    if partitions is None:
        collectedmacros = collectmacrosforrandompartitions(grph, labels, labeldict, pairs, trns, ncls, rng, itercount, valshare, testshare)
    else: 
        collectedmacros = collectmacrosforstoredpartitions(grph, labels, labeldict, pairs, trns, ncls, rng, partitions, conseq)
    result = {}
    if debug:
        print("====================================")
        print("+++++Results for "+ datafile+ "+++++")
    for feature in collectedmacros:
        if debug:
            print(f"{feature} f1 macro mean: {np.average(collectedmacros[feature]):.4f} std: {np.std(collectedmacros[feature]):.4f}" )
        result[feature] = {"mean": np.average(collectedmacros[feature]), 
                           "std": np.std(collectedmacros[feature]), 
                           "values":collectedmacros[feature] }
    return result
        
    
def runheuristics(workdir, infile, rng):
    with open(os.path.join(workdir, infile),"r") as f:
        explist = json.load(f)
    
    result = {}
    for dataset in explist:
        print("Running experiments for", dataset)
        datasetexplist = explist[dataset]
        datasetresults = []
        for exp in datasetexplist:
            
            datafile = os.path.join(workdir, exp["datafile"])
            
            with open(os.path.join(workdir, exp["partitionfile"]),"r") as f:
                partitions = json.load(f)
            
            runresult = run(datafile, partitions=partitions["partitions"], conseq=False, debug=False, filter_params = None, rng =rng)
            datasetresults.append(runresult)
        result[dataset] = datasetresults
    return result
    
def runandsaveheuristics(workdir, infile, outfile, rng): 
    result = runheuristics(workdir, infile, rng)
    with open(os.path.join(workdir, outfile),"w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, sort_keys=True)
   
    
if __name__=="__main__":
    valshare = 0.2
    testshare = 0.2
    itercount = 100
    
    rng = np.random.default_rng(2023)
    datapath = "../datasets/"
    dataset1fname = datapath+"dataset.csv"
    #dataset1fname = datapath+"Western-Europe_weights_partial_labels.csv"
    # test1: random partitions
    print("\n test1: generated partitions")
    run(dataset1fname, valshare, testshare, itercount, rng=rng)