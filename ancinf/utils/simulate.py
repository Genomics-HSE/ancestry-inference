import json
import os
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import os.path
import time, datetime
from contextlib import ExitStack


from .genlink import simulate_graph_fn, generate_matrices_fn, independent_test
from .baseheuristic import getprobandmeanmatrices, composegraphs, checkpartition, combinationgenerator
from .baseheuristic import getrandompermutation, dividetrainvaltest, gettrainvaltestnodes
from .runheuristic import translateconseqtodf, runheuristics
from .runheuristic import run as runheur
from .ibdloader import load_pure, translate_indices


import torch
import numpy as np
import random
import sys
from sklearn.metrics import f1_score
import shutil

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(''), os.path.pardir)))
from .genlink import DataProcessor, BaselineMethods, NullSimulator, Trainer,  TAGConv_3l_128h_w_k3, \
                  TAGConv_3l_512h_w_k3, GINNet, AttnGCN, TAGConv_9l_128h_k3, GCNConv_3l_128h_w, \
                  TAGConv_9l_512h_nw_k3, MLP_3l_128h, MLP_3l_512h, MLP_9l_128h,\
                  MLP_9l_512h, \
GINNet_narrow_short, GINNet_wide_short, GINNet_narrow_long, GINNet_wide_long, \
AttnGCN_narrow_short, AttnGCN_wide_short, AttnGCN_narrow_long,  AttnGCN_wide_long
    
NNs = {
    "MLP_3l_128h": MLP_3l_128h,
    "MLP_3l_512h": MLP_3l_512h,
    "MLP_9l_128h": MLP_9l_128h,
    "MLP_9l_512h": MLP_9l_512h,
    "TAGConv_3l_128h_w_k3": TAGConv_3l_128h_w_k3,
    "TAGConv_3l_512h_w_k3": TAGConv_3l_512h_w_k3,    
    "TAGConv_9l_512h_nw_k3": TAGConv_9l_512h_nw_k3,
    "TAGConv_9l_128h_k3": TAGConv_9l_128h_k3,
    "TAGConv_3l_128h_w_k3_gb": TAGConv_3l_128h_w_k3,
    "TAGConv_3l_512h_w_k3_gb": TAGConv_3l_512h_w_k3,    
    "TAGConv_9l_512h_nw_k3_gb": TAGConv_9l_512h_nw_k3,
    "TAGConv_9l_128h_k3_gb": TAGConv_9l_128h_k3,
    "GCNConv_3l_128h_w": GCNConv_3l_128h_w,
    "GCNConv_3l_128h_w_gb": GCNConv_3l_128h_w,
    "GINNet": GINNet,
    "GINNet_narrow_short": GINNet_narrow_short, 
    "GINNet_wide_short": GINNet_wide_short, 
    "GINNet_narrow_long": GINNet_narrow_long, 
    "GINNet_wide_long": GINNet_wide_long,
    "GINNet_gb": GINNet,
    "GINNet_narrow_short_gb": GINNet_narrow_short, 
    "GINNet_wide_short_gb": GINNet_wide_short, 
    "GINNet_narrow_long_gb": GINNet_narrow_long, 
    "GINNet_wide_long_gb": GINNet_wide_long,
    "AttnGCN": AttnGCN,
    "AttnGCN_narrow_short": AttnGCN_narrow_short, 
    "AttnGCN_wide_short": AttnGCN_wide_short,
    "AttnGCN_narrow_long": AttnGCN_narrow_long, 
    "AttnGCN_wide_long":AttnGCN_wide_long,
    "AttnGCN_gb": AttnGCN,
    "AttnGCN_narrow_short_gb": AttnGCN_narrow_short, 
    "AttnGCN_wide_short_gb": AttnGCN_wide_short,
    "AttnGCN_narrow_long_gb": AttnGCN_narrow_long, 
    "AttnGCN_wide_long_gb":AttnGCN_wide_long
}



def labeldict_to_labellist(labeldict):
    count = len(labeldict)
    result = [""] * count
    for lbl in labeldict:
        idx = labeldict[lbl]
        result[idx] = lbl
    return result

def collectdatasetparams(fnamepath, filters={}):
    pairs, weights, labels, labeldict, idxtranslator = load_pure( fnamepath, debug=False, **filters )
    #print("load ok!")
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

    result = {
        "pop_names": labellist,
        "pop_sizes": pop_sizes,
        "edge_probability": probslist,
        "mean_weight":meanslist
    }
    return result
    


def collectparams(datadir, workdir, infile):
    '''
    get parameters for every dataset in metafile
    '''
    
    with open(join(workdir, infile),"r") as f:
        meta = json.load(f)
    
    paramdict = {"datasets": {}}
    for k in ["experiments", "simulator", "crossvalidation"]:
        if k in meta:
            paramdict[k] = meta[k]

    datasets = meta["datasets"]
    
    for datasetname in datasets:        
        print("processing dataset", datasetname)
        fnamepath = join(datadir, datasetname+'.csv')        
        filters = datasets[datasetname]["filters"]
        dsparams = collectdatasetparams(fnamepath, filters)
        
        
        paramdict["datasets"][datasetname] = dsparams
        
    #print(paramdict)    
    return paramdict

    
def collectandsaveparams(datadir, workdir, infile, outfile):
    print(f"Collecting parameters for datasets from {datadir}")
    collected = collectparams(datadir, workdir, infile)    
    outpathfile = os.path.join(workdir, outfile)
    with open(outpathfile,"w", encoding="utf-8") as f:
        json.dump(collected, f, indent=4, sort_keys=True)

        

        

def simulateandsaveonedataset(datasetparams, offset, fname, rng):        
    population_sizes = datasetparams["pop_sizes"]        
    edge_probs = np.array(datasetparams["edge_probability"])
    mean_weight = np.array(datasetparams["mean_weight"]) - offset #the next function wants corrected mean weights
    classes = datasetparams["pop_names"]
    #print("Edge probs:",edge_probs)
    #print("Mean weights:", mean_weight)
    #print("Population sizes:", population_sizes)
    counts, means, pop_index = generate_matrices_fn(population_sizes, offset, edge_probs, mean_weight, rng) 
    simulate_graph_fn(classes, means, counts, pop_index, fname)
        
def updateparams(datasetparams, experiment):
    resultparams = {}
    for k in datasetparams:
        resultparams[k] = datasetparams[k]
    #print(experiment)
    new_mean_weight = np.array(datasetparams["mean_weight"])
    new_edge_probability = np.array(datasetparams["edge_probability"])
    
    new_mean_weight = new_mean_weight * experiment["all_weight_scale"]
    new_edge_probability = new_edge_probability * experiment["all_edge_probability_scale"]
    
    for idm in range(new_mean_weight.shape[0]):
        for idn in range(new_mean_weight.shape[0]):
            if idm == idn:
                new_mean_weight[idm, idn] *= experiment["diag_weight_scale"]
                new_edge_probability[idm, idn] *= experiment["diag_edge_probability_scale"]
            else:
                new_mean_weight[idm, idn] *= experiment["nondg_weight_scale"]
                new_edge_probability[idm, idn] *= experiment["nondg_edge_probability_scale"]
    
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
    
    

def simulateandsave(workdir, infile, outfile, rng):
    position = infile.find('.params')
    if position>0:
        projname = infile[:position]
    else:
        projname = infile    
    
    infilepath = os.path.join(workdir, infile)
    
    print(f"Running simulations for parameters from {infile}")
    with open(infilepath, 'r') as f:
        dct = json.load(f)  
    experiments = dct["experiments"]
    datasets = dct["datasets"]
    simparams = dct["simulator"]
    offset = simparams["offset"]
    trainparams = dct["crossvalidation"]
    
    expfiledict = {}   
    #iterate through datasets
    for datasetname in datasets:
        experimentlist = []
        #now iterate through experiments               
        #print(experiments)
        expgen = combinationgenerator(experiments)
        datasetparams = datasets[datasetname]
            
        for expnum, experiment in enumerate(expgen):
            #print(experiment)
            datafilename =  projname+'_'+datasetname+'_exp'+str(expnum)+'.csv' 
            partfilename =  projname+'_'+datasetname+'_exp'+str(expnum)+'.split' 
            
            updateddatasetparams = updateparams(datasetparams, experiment)
            simulateandsaveonedataset(updateddatasetparams, offset, os.path.join(workdir, datafilename), rng)
            
            savepartitions(os.path.join(workdir, datafilename), trainparams["valshare"], trainparams["testshare"], trainparams["split_count"], 
                           os.path.join(workdir, partfilename), rng)
            
            expdict = { "id":expnum,
                       "experiment":experiment,
                       "datasetparams":datasetparams,
                       "datafile":datafilename,
                       "crossvalidation": trainparams,
                       "splitfile":partfilename
            }
            experimentlist.append(expdict)            
        expfiledict[datasetname] = experimentlist 
        
    experimentlistfile = os.path.join(workdir, outfile )
    #create file with all the experiments descritption
    with open(experimentlistfile,"w") as f:
        json.dump(expfiledict, f, indent=4, sort_keys=True)

def filterandsaveonedataset(indatafilename, outdatapathfilenamebase, filters, cleanshare, maskshare, rng):
    retval = None
    pairs, weights, labels, labeldict, idxtranslator = load_pure( indatafilename, debug=False, **(filters))
    conseq_pairs = translate_indices(pairs, idxtranslator)
    labellist = labeldict_to_labellist(labeldict)
        
    graphdata = composegraphs(pairs, weights, labels, labeldict, idxtranslator)
    ncls = graphdata[0]['nodeclasses']
    grph = graphdata[0]['graph']
    trns = graphdata[0]['translation']
    #translate indices in ncls to original indices
    permt = getrandompermutation(ncls, rng)

    trainnodeclasses, valnodeclasses, testnodeclasses = dividetrainvaltest(ncls, maskshare, cleanshare, permt)
    part_ok, part_errors = checkpartition(grph, trainnodeclasses, valnodeclasses, testnodeclasses, details=False, trns=trns)
    if not part_ok:
        print("bad partition for normal-mask-clean dataset part")                    
    normalnodes, masknodes, cleannodes = gettrainvaltestnodes(trainnodeclasses, valnodeclasses, testnodeclasses)    
    print("normalnodes", idxtranslator[normalnodes].tolist())
    if maskshare>0:
        print("masknodes", idxtranslator[masknodes].tolist())
    if cleanshare > 0:
        print("cleannodes", idxtranslator[cleannodes].tolist())
    retval = {}
    if cleanshare > 0:
        clnodes = idxtranslator[cleannodes].tolist()
        #print("clean nodes:", clnodes)
        cllabels = [labellist[lbl] for lbl in labels[cleannodes] ] 
        #print("clean node labels:", cllabels)
        retval["cleannodes"] = clnodes
        retval["cleannodelabels"] = cllabels
       
    if maskshare > 0:
        msnodes = idxtranslator[masknodes].tolist()        
        retval["maskednodes"] = msnodes
        
    with ExitStack() as filestack:
        f = filestack.enter_context(open(outdatapathfilenamebase+'.csv', 'w', encoding="utf-8"))
        f.write('node_id1,node_id2,label_id1,label_id2,ibd_sum,ibd_n\n')
        if cleanshare > 0:
            f_clean = filestack.enter_context(open(outdatapathfilenamebase+'_clean.csv', 'w', encoding="utf-8"))
            f_clean.write('node_id1,node_id2,label_id1,label_id2,ibd_sum,ibd_n\n')
        if maskshare > 0:
            f_masked = filestack.enter_context(open(outdatapathfilenamebase+'_masked.csv', 'w', encoding="utf-8"))
            f_masked.write('node_id1,node_id2,label_id1,label_id2,ibd_sum,ibd_n\n')
            f_removed = filestack.enter_context(open(outdatapathfilenamebase+'_removed.csv', 'w', encoding="utf-8"))
            f_removed.write('node_id1,node_id2,label_id1,label_id2,ibd_sum,ibd_n\n')
        
        for idx, pair in enumerate(pairs):
            i = conseq_pairs[idx][0]
            j = conseq_pairs[idx][1]                    
            label_i = labellist[labels[i]]
            label_j = labellist[labels[j]]

            name_j = masked_name_j = label_j if "," not in label_j else '\"' + label_j + '\"'                
            name_i = masked_name_i = label_i if "," not in label_i else '\"' + label_i + '\"'
            if maskshare > 0:
                if i in masknodes:
                    masked_name_i = "masked"                
                if j in masknodes:
                    masked_name_j = "masked"

            if ((i in normalnodes) or (i in masknodes)) and ((j in normalnodes) or (j in masknodes)):                        
                f.write(f'node_{pair[0]},node_{pair[1]},{name_i},{name_j},{weights[idx][0]},{pair[2]}\n')
                if maskshare > 0:
                    f_masked.write(f'node_{pair[0]},node_{pair[1]},{masked_name_i},{masked_name_j},{weights[idx][0]},{pair[2]}\n')
                    if (i in normalnodes) and (j in normalnodes):                        
                        f_removed.write(f'node_{pair[0]},node_{pair[1]},{name_i},{name_j},{weights[idx][0]},{pair[2]}\n')
            else:
                if i in cleannodes:                            
                    if j in cleannodes:
                        #we do not want edges between clean nodes
                        pass
                    else:
                        f_clean.write(f'node_{pair[0]},node_{pair[1]},unknown,{masked_name_j},{weights[idx][0]},{pair[2]}\n')
                elif j in cleannodes:
                    f_clean.write(f'node_{pair[0]},node_{pair[1]},{masked_name_i},unknown,{weights[idx][0]},{pair[2]}\n')
                else:
                    print("ERROR! No node from pair belong to train or clean partitions")
        
    return retval
    
        
def preprocess(datadir, workdir, infile, outfile, rng):
    position = infile.find('.ancinf')
    if position>0:
        projname = infile[:position]
    else:
        projname = infile    
        
    infilepath = os.path.join(workdir, infile)
    
    print(f"Preprocessing {infile}")
    with open(infilepath, 'r') as f:
        dct = json.load(f)          
    datasets = dct["datasets"]
    trainparams = dct["crossvalidation"]
    
    expfiledict = {}   
    #iterate through datasets
    for datasetname in datasets:
        print("processing dataset", datasetname)
        experimentlist = []        
        datasetparams = datasets[datasetname]
                
        indatafilename = os.path.join(datadir, datasetname+'.csv' )
        outdatafilenamebase =  projname+'_'+datasetname                
        
        if "cleanshare" in trainparams:
            cleanshare = trainparams["cleanshare"]            
        else:
            cleanshare = 0
            
        if "maskshare" in trainparams:
            maskshare = trainparams["maskshare"]            
        else:
            maskshare = 0
            
            
        retval = filterandsaveonedataset(indatafilename, os.path.join(workdir, outdatafilenamebase), datasetparams["filters"], cleanshare, maskshare, rng)
        
        savepartitions(os.path.join(workdir, outdatafilenamebase+ '.csv'), trainparams["valshare"], trainparams["testshare"],
                       trainparams["split_count"], os.path.join(workdir, outdatafilenamebase+ '.split'), rng)
        if "maskshare" in trainparams:
            savepartitions(os.path.join(workdir, outdatafilenamebase+ '_removed.csv'), trainparams["valshare"], trainparams["testshare"],
                       trainparams["split_count"], os.path.join(workdir, outdatafilenamebase+ '_removed.split'), rng)
                                                     
        expdict = {
                   "datafile":outdatafilenamebase+'.csv',
                   "crossvalidation": trainparams,
                   "splitfile":outdatafilenamebase+'.split'
        }
        if "cleanshare" in trainparams:
            expdict["cleanfile"] = outdatafilenamebase+'_clean.csv'
            expdict["cleannodes"] =  retval["cleannodes"]
            expdict["cleannodelabels"] = retval["cleannodelabels"]
        if "maskshare" in trainparams:
            expdict["maskednodes"] = retval["maskednodes"]
            expdict["maskedfile"] = outdatafilenamebase+'_masked.csv'
            expdict["removedfile"] = outdatafilenamebase+'_removed.csv'
            expdict["removedsplitfile"] = outdatafilenamebase+'_removed.split'
        experimentlist.append(expdict)            
        expfiledict[datasetname] = experimentlist 
        
    experimentlistfile = os.path.join(workdir, outfile )
    #create file with all the experiments descritption
    with open(experimentlistfile,"w", encoding="utf-8") as f:
        json.dump(expfiledict, f, indent=4, sort_keys=True)
            

    
def getexplistinfo(explist):
    datasetcount = len(explist)
    for dsname in explist:
        ds = explist[dsname]
        expcount = len(ds)
        splitcount = ds[0]["crossvalidation"]["split_count"]
        mlpcount = len(ds[0]["crossvalidation"]["mlps"])
        gnncount = len(ds[0]["crossvalidation"]["gnns"])
        heucount = len(ds[0]["crossvalidation"]["heuristics"])
        comdetcount = len(ds[0]["crossvalidation"]["community_detection"])
        break
       
    totalruncount = datasetcount * expcount * splitcount
    print (f"Total runs: {totalruncount} = {datasetcount} datasets x {expcount} parameter values x {splitcount} splits")
    print (f"Every run: {heucount} heuristics, {comdetcount} community detection algorithms, {mlpcount} fully connected NNs, {gnncount} graph NNs")
    
    return totalruncount, expcount


def processpartition_nn(expresults, datafile, partition, maskednodes, gnnlist, mlplist, comdetlist, fullist, runidx, runbasename, log_weights, gpuidx):
    print(f"{datafile=}")
    print(f"{maskednodes=}")
    print(f"{partition=}")
    #TODO maybe it will not hurt to create DataProcessor once for a split 
    #(now they are created for every classifier from scratch)
    train_list = []
    val_list = []
    test_list = []
    for popl in partition["train"]:
        train_list = train_list + partition["train"][popl]   
        val_list = val_list + partition["val"][popl]   
        test_list = test_list + partition["test"][popl]   

    train_split = np.array(train_list)
    valid_split = np.array(val_list)
    test_split = np.array(test_list)
    if not(maskednodes is None):
        maskednodes = np.array(maskednodes)
    
    #gnn
    for gnnclass in gnnlist:
        run_name = runbasename + "_"+gnnclass
        if type(gnnlist) is dict:
            model_params = gnnlist[gnnclass]
        else:
            model_params = {}
        print("NEW RUN:", run_name)
        starttime = time.time()
        runresult = simplified_genlink_run(datafile, train_split, valid_split, test_split, run_name, gnnclass, gnn=True, logweights=log_weights, gpu=gpuidx, maskednodes=maskednodes, model_params = model_params)
        runtime = time.time() - starttime
        runresult["time"] = runtime        
        print("RUN COMPLETE!", gnnclass, runresult)        
        expresults[gnnclass].append(runresult)

    #mlp
    for mlpclass in mlplist:
        run_name = runbasename +"_"+mlpclass
        if type(gnnlist) is dict:
            model_params = gnnlist[gnnclass]
        else:
            model_params = {}
        print("NEW RUN:", run_name)
        starttime = time.time()
        runresult = simplified_genlink_run(datafile, train_split, valid_split, test_split, run_name, mlpclass, gnn=False, logweights=log_weights, gpu=gpuidx, maskednodes=maskednodes, model_params = model_params )
        runtime = time.time() - starttime
        runresult["time"] = runtime        
        print("RUN COMPLETE!", mlpclass, runresult)
        expresults[mlpclass].append(runresult)

    #community detection
    if comdetlist!=[]:        
        for comdet in comdetlist:
            starttime = time.time()
            if comdet == "Spectral":
                dp = DataProcessor(datafile)
                dp.load_train_valid_test_nodes(train_split, valid_split, test_split, 'numpy')        
                bm = BaselineMethods(dp)                
                score = bm.spectral_clustering()                
            if comdet == "Agglomerative":
                dp = DataProcessor(datafile)
                dp.load_train_valid_test_nodes(train_split, valid_split, test_split, 'numpy')        
                bm = BaselineMethods(dp)                
                score = bm.agglomerative_clustering()                
            if comdet == "Girvan-Newmann":
                dp = DataProcessor(datafile)
                dp.load_train_valid_test_nodes(train_split, valid_split, test_split, 'numpy')        
                bm = BaselineMethods(dp)                
                score = bm.girvan_newman()  
            if comdet == "LabelPropagation":
                dplp = DataProcessor(datafile)
                dplp.load_train_valid_test_nodes(train_split, valid_split, test_split, 'numpy')        
                dplp.make_train_valid_test_datasets_with_numba('one_hot', 'homogeneous', 'multiple', 'multiple', 'tmp')
                bmlp  = BaselineMethods(dplp)                
                score = bmlp.torch_geometric_label_propagation(1, 0.0001)  
            if comdet == "RelationalNeighbor":
                dp = DataProcessor(datafile)
                dp.load_train_valid_test_nodes(train_split, valid_split, test_split, 'numpy') 
                bm = BaselineMethods(dp)
                score = bm.relational_neighbor_classifier(0.001)        
            if comdet == "MultiRankWalk":
                dp = DataProcessor(datafile)
                dp.load_train_valid_test_nodes(train_split, valid_split, test_split, 'numpy') 
                bm = BaselineMethods(dp)
                score = bm.multi_rank_walk(0.8)      
            if comdet == "RidgeRegression":
                dp = DataProcessor(datafile)
                dp.load_train_valid_test_nodes(train_split, valid_split, test_split, 'numpy') 
                bm = BaselineMethods(dp)
                score = bm.ridge_regression(0.5, 3)        
                
                    
            runtime = time.time() - starttime
            print(comdet,":", score)
            runresult = score
            runresult["time"] = runtime            
            expresults[comdet].append(runresult)

            
            
def inference(workdir, traindf, inferdf, model, model_weights, gpu=0):
    #1. get ids of every unknown node         
    inferredlabels = []
    
    dfmain = pd.read_csv(traindf)
    dfclean = pd.read_csv(inferdf)    
    #find them
    pairs, weights, labels, labeldict, idxtranslator = load_pure( inferdf, debug=False)
    unklblidx = labeldict["unknown"]
    cleannodes = list( idxtranslator[ labels == unklblidx ] )
    print("unknown nodes:", cleannodes)
    print("weights path:", model_weights)    
    for node in cleannodes:
        print("infering class for node", node)
        fltr1 = dfclean[dfclean["node_id1"] =="node_" +str(node)]                    
        fltr2 = dfclean[dfclean["node_id2"] =="node_" +str(node)]                                        
        onenodedf = pd.concat([dfmain, fltr1, fltr2])                    
        df = onenodedf.reset_index(drop=True)                         
        #TODO fix test_type depending on the network
        testresult = independent_test(model_weights, NNs[model], df, node, gpu, test_type='one_hot' )        
        print("inference results", testresult)
        inferredlabels.append( testresult )
    
    return cleannodes, inferredlabels
    
           

def runcleantest(cleanexpresults, cleannodes, cleannodelabels, cleantestdataframes, gnnlist, run_base_name, gpu):
    for nnclass in gnnlist:
        run_name = os.path.join(run_base_name + "_"+nnclass, "model_best.bin" )
        inferredlabels = []
        for node in cleannodes:
            print("infering class for node", node)
            print(cleantestdataframes[node])
            #cleantestdataframes[node].to_csv("temp.csv")
            #TODO fix test_type depending on the network
            testresult = independent_test(run_name, NNs[nnclass], cleantestdataframes[node], node, gpu, test_type='one_hot' )            
            print("clean test classification", testresult)
            inferredlabels.append( testresult )
        runresult = f1_score(cleannodelabels, inferredlabels, average='macro')
        cleanexpresults[nnclass].append(runresult)

def getbrief(fullres):
    briefres = {}
    for dataset in fullres: 
        briefres[dataset]=[]
        for exp in fullres[dataset]: #fullres[dataset] is a list of the ds exps
            expres = { "dataset_begin": exp["dataset_begin"],
                       "dataset_end": exp["dataset_end"],
                       "dataset_time": exp["dataset_time"],
                       "exp_idx": exp["exp_idx"],
                       "classifiers":{}
                     }
            for classifier in exp["classifiers"]:
                expres["classifiers"][classifier] = {}
                if "f1_macro" in exp["classifiers"][classifier]:
                    expres["classifiers"][classifier]["f1_macro_mean"] = exp["classifiers"][classifier]["f1_macro"]["mean"]
                    if "clean_mean" in exp["classifiers"][classifier]["f1_macro"]:
                        expres["classifiers"][classifier]["f1_macro_mean_clean"] = exp["classifiers"][classifier]["f1_macro"]["clean_mean"]
                
            briefres[dataset].append(expres)
        
    return briefres
        
def compiledsresults(expresults, fullist):
    dsres = {}
    #dsbrief = {}
    for nnclass in fullist:
        dsres[nnclass] = {}
        #dsbrief[nnclass] = {}
        if expresults[nnclass]!=[]:
            metrics = expresults[nnclass][0]            
            for metric in metrics:
                if metric == "class_scores":
                    dsres[nnclass]["class_scores"] = {}
                    for cl in expresults[nnclass][0]["class_scores"]:
                        dsres[nnclass]["class_scores"][cl] = {} 
                        dsres[nnclass]["class_scores"][cl]["values"]=[]
                else:
                    dsres[nnclass][metric]={}
                    dsres[nnclass][metric]["values"] = []                
                
            for splitresult in expresults[nnclass]:                
                for metric in metrics:
                    if metric == "class_scores":
                        for cl in splitresult["class_scores"]:
                            dsres[nnclass]["class_scores"][cl]["values"].append(splitresult["class_scores"][cl])
                    else:
                        dsres[nnclass][metric]["values"].append(splitresult[metric])
            #now we have all values arrays filled
            for metric in metrics:
                if metric == "class_scores":
                    for cl in expresults[nnclass][0]["class_scores"]:
                        dsres[nnclass][metric][cl]["mean"] = np.average(dsres[nnclass][metric][cl]["values"])
                        dsres[nnclass][metric][cl]["std"] = np.std(dsres[nnclass][metric][cl]["values"])
                else:
                    dsres[nnclass][metric]["mean"] = np.average(dsres[nnclass][metric]["values"])
                    dsres[nnclass][metric]["std"] = np.std(dsres[nnclass][metric]["values"])            
            #for metric in ["f1_macro", "time"]:                
                #dsbrief[nnclass][metric] = dsres[nnclass][metric]["mean"]
    return dsres#, dsbrief



def runandsaveall(workdir, infile, outfilebase, fromexp, toexp, fromsplit, tosplit, gpu):
    #loop through dataset
    #  loop through experiments (different
    #    loop though splits
    #      loop through classifiers
    #        train test update results resave with new data    
    with open(os.path.join(workdir, infile),"r") as f:
        explist = json.load(f)    
    totalruncount, expcount = getexplistinfo(explist)
    
    no_exp_postfix = (fromexp is None) and (toexp is None)
    if fromexp is None:
        fromexp = 0
    else:
        fromexp = int(fromexp)
    if toexp is None:
        toexp = expcount
    else:
        toexp = int(toexp)
    
    if no_exp_postfix:
        outfile_exp_postfix = ""
    else:
        outfile_exp_postfix = "_e" + str(fromexp) +"-"+ str(toexp)    
    runfolder = outfilebase+outfile_exp_postfix  + "_runs"
    
    
    print(f"We will process experiments from [{fromexp} to {toexp}) on gpu {gpu}")
    
    runidx = 1
    
    result = {}
    for dataset in explist:
        print("Running experiments for", dataset)
        datasetexplist = explist[dataset]
        datasetresults = []
        for exp_idx in range(fromexp, toexp):            
            exp = datasetexplist[exp_idx]
            
            if "maskednodes" in exp:
                datafile = os.path.join(workdir, exp["maskedfile"])
            else:
                datafile = os.path.join(workdir, exp["datafile"])
                
                
            
            with open(os.path.join(workdir, exp["splitfile"]),"r") as f:
                partitions = json.load(f)
            
            total_split_count = exp["crossvalidation"]["split_count"] 
            
            no_split_postfix = (fromsplit is None) and (tosplit is None)
            if fromsplit is None:
                dsfromsplit = 0
            else:
                dsfromsplit = int(fromsplit)
            if tosplit is None:
                dstosplit = total_split_count
            else:
                dstosplit = int(tosplit)
            if no_split_postfix:
                outfile_split_postfix = ""
            else:
                outfile_split_postfix = "_s" + str(dsfromsplit) +"-"+ str(dstosplit)
            outfile = outfilebase+outfile_exp_postfix+outfile_split_postfix+'.results'
            
            
            log_weights = exp["crossvalidation"]["log_weights"] 
            heurlist = exp["crossvalidation"]["heuristics"] 
            comdetlist = exp["crossvalidation"]["community_detection"]
            mlplist = exp["crossvalidation"]["mlps"]
            gnnlist = exp["crossvalidation"]["gnns"]            
            fullist = []
            if type(heurlist) is list:
                fullist = fullist + heurlist
            else:
                fullist = fullist + list(heurlist.keys())
            if type(comdetlist) is list:
                fullist = fullist + comdetlist
            else:
                fullist = fullist + list(comdetlist.keys())
            if type(mlplist) is list:
                fullist = fullist + mlplist
            else:
                fullist = fullist + list(mlplist.keys())
            if type(gnnlist) is list:
                fullist = fullist + gnnlist
            else:
                fullist = fullist + list(gnnlist.keys())
            
            expresults = {nnclass:[] for nnclass in fullist} 
            datasetresults.append(expresults) #({nnclass: {"mean": -1, "std": -1, "values":[]} for nnclass in fullist})
            
            datasetstart = datetime.datetime.now().strftime("%H:%M on %d %B %Y")
            datasetstarttime = time.time()
            #1. all heuristics for all partitions at once
            if len(heurlist)>0 and (dsfromsplit==0):
                #if we have masked labels, we should use dataset and split with them removed
                starttime = time.time()
                if "maskednodes" in exp:
                    heurdatafile = os.path.join(workdir, exp["removedfile"])
                    with open(os.path.join(workdir, exp["removedsplitfile"]),"r") as f:
                        rempartitions = json.load(f)
                    heurpartitions = rempartitions["partitions"]
                else:
                    heurdatafile = datafile
                    heurpartitions = partitions["partitions"]
                heuresult = runheur(heurdatafile, partitions=heurpartitions, conseq=False, debug=False, filter_params = None)
                runtime = time.time() - starttime                
                for heurclass in heurlist: 
                    expresults[heurclass] = [ {"f1_macro": res, 
                                               "time": runtime/len(heuresult[heurclass]["values"]) 
                                              } for res in heuresult[heurclass]["values"] ]
            
            #2. GNNs and MLPs partition by partition
            #if clean test is requiered we prepare dataframes with just one unlabelled node
            if "cleanfile" in exp:
                #prepare dataframes once for all neuronetworks
                #compose df from dfmain and edges to node from dfclean
                cleanfile = os.path.join(workdir, exp["cleanfile"])
                dfmain = pd.read_csv(datafile)
                dfclean = pd.read_csv(cleanfile)
                cleantestdataframes = {}
                cleannodes = exp["cleannodes"]            
                cleannodelabels = exp["cleannodelabels"]
                for node in cleannodes:
                    fltr1 = dfclean[dfclean["node_id1"] =="node_" +str(node)]                    
                    fltr2 = dfclean[dfclean["node_id2"] =="node_" +str(node)]                                        
                    onenodedf = pd.concat([dfmain, fltr1, fltr2])                    
                    cleantestdataframes[node] = onenodedf.reset_index(drop=True)
                cleanexpresults = {nnclass:[] for nnclass in gnnlist} 
             
            if "maskednodes" in exp:
                maskednodes = exp["maskednodes"]
            else:
                maskednodes = None

            #begin partition loop
            for part_idx in range(dsfromsplit, dstosplit):
                partition = partitions["partitions"][part_idx]
                print(f"=========== Run {runidx} of {totalruncount} ======================")
                run_base_name = os.path.join(workdir, runfolder, "run_"+dataset+"_exp"+str(exp_idx)+"_split"+str(part_idx))
                processpartition_nn(expresults, datafile, partition, maskednodes, gnnlist, mlplist, comdetlist, fullist, runidx, run_base_name, log_weights, gpu )
                fullres = compiledsresults(expresults, fullist)                
                datasetfinish = datetime.datetime.now().strftime("%H:%M on %d %B %Y")
                datasetfinishtime = time.time()
                datasetresults[-1] = {"dataset_begin": datasetstart, "dataset_end": datasetfinish, 
                                      "dataset_time": datasetfinishtime - datasetstarttime,
                                      "exp_idx": exp_idx, "classifiers": fullres } 
                
                result[dataset] = datasetresults
                
                with open(os.path.join(workdir, outfile+'.incomplete'),"w", encoding="utf-8") as f:
                    #print(result)
                    json.dump({"brief": getbrief(result), "details":result}, f, indent=4, sort_keys=True)  
                
                runidx+=1
                #now clean test if requested
                if ("cleanfile" in exp) and (gnnlist != []):
                    print("Running clean inference test")                
                    runcleantest(cleanexpresults, cleannodes, cleannodelabels, cleantestdataframes, gnnlist, run_base_name, gpu)  
                    for nnclass in cleanexpresults:
                        datasetresults[-1]["classifiers"][nnclass]["f1_macro"].update({"clean_mean": np.average(cleanexpresults[nnclass]), 
                                             "clean_std": np.std(cleanexpresults[nnclass]), 
                                             "clean_values":cleanexpresults[nnclass]} )                                               
                    result[dataset] = datasetresults
                    with open(os.path.join(workdir, outfile+'.incomplete'),"w", encoding="utf-8") as f:
                        json.dump({"brief": getbrief(result), "details":result}, f, indent=4, sort_keys=True)
            #end partition loop
    shutil.move(os.path.join(workdir, outfile+'.incomplete'), os.path.join(workdir, outfile))
    return os.path.join(workdir, outfile)       
            
            
def simplified_genlink_run(dataframe_path, train_split, valid_split, test_split, rundir, nnclass, gnn=True, logweights=False, gpu=0, maskednodes=None, model_params={}):
    '''
        returns f1 macro for one experiment
    '''
    dp = DataProcessor(dataframe_path)
    dp.load_train_valid_test_nodes(train_split, valid_split, test_split, 'numpy', mask_path = maskednodes)
    masking = not (maskednodes is None)
    if gnn:        
        if nnclass[-3:] == '_gb': 
            features = 'graph_based'
            train_dataset_type = "one"
        else:
            features = 'one_hot'
            train_dataset_type = 'multiple'
        dp.make_train_valid_test_datasets_with_numba(features, 'homogeneous', train_dataset_type, 'multiple', rundir, log_edge_weights=logweights, masking = masking)    
        trainer = Trainer(dp, NNs[nnclass], 0.0001, 5e-5, torch.nn.CrossEntropyLoss, 10, rundir, 2, 20,
                      features, 50, 10, cuda_device_specified=gpu, masking = masking, model_params = model_params)
    else:#mlp        
        dp.make_train_valid_test_datasets_with_numba('graph_based', 'homogeneous', 'one', 'multiple', rundir, log_edge_weights=logweights, masking=masking)    
        trainer = Trainer(dp, NNs[nnclass], 0.0001, 5e-5, torch.nn.CrossEntropyLoss, 10, rundir, 2, 50,
                      'graph_based', 50, 10, cuda_device_specified=gpu, masking = masking, model_params = model_params)
    
    return trainer.run()           


def getplotdata(explistfile, resultfile, parameter):
    '''
         combine experiment results into data usable for plotting
         
    Parameters
    ----------
    explistfile: str
        .expfile filename
    resultsfile: str
        .result filename
    parameter: str
        parameter name that is varied across experiments
        
    Returns
    -------
    results: dict
        for every dataset we have a dict of 'x's (parameter values)
        param name and classifier mean
    '''
    with open(explistfile,'r') as f:
        explist = json.load(f)
    with open(resultfile,'r') as f:
        res = json.load(f)

    plotdata = {}
    for dataset in explist:
        print(dataset)
        datasetdata = {classifier:[] for classifier in res[dataset][0]}
        datasetdata["param"] = parameter
        datasetdata["x"] = []
        for expidx in range(len(explist[dataset])):
            #print("experiment", expidx)
            exp = explist[dataset][expidx]
            result = res[dataset][expidx]
            datasetdata["x"].append(exp["experiment"][parameter])
            for classifier in result:
                datasetdata[classifier].append(result[classifier]['mean'])

        plotdata[dataset] = datasetdata
    return plotdata    

def plotclassifierdependency(plotdata, classifierlist):
    '''
         Plot selected classifier quality dependence on parameter
         
    Parameters
    ----------
    plotdata: dict
        data produced by getplotdata
    classifierlist: list
        specify classifiers to plot their quality
        
    Returns
    -------
    
    '''
    from matplotlib import pyplot as plt
    plt.rcParams["figure.figsize"] = (14,7)
    for dataset in plotdata:
        for classifier in classifierlist:
            plt.plot(plotdata[dataset]['x'], plotdata[dataset][classifier], label=classifier)
    
        plt.title(dataset+": "+plotdata[dataset]["param"])
        plt.legend()
        plt.show()


            
        
if __name__=="__main__":
    print("just a test")