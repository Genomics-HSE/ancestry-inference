import json
import os
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import os.path



from .genlink import simulate_graph_fn, generate_matrices_fn, independent_test
from .baseheuristic import getprobandmeanmatrices, composegraphs, checkpartition, combinationgenerator
from .baseheuristic import getrandompermutation, dividetrainvaltest, gettrainvaltestnodes
from .runheuristic import translateconseqtodf, runheuristics
from .ibdloader import load_pure, translate_indices


import torch
import numpy as np
import random
import sys
from sklearn.metrics import f1_score

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(''), os.path.pardir)))
from .genlink import DataProcessor, NullSimulator, Trainer,  TAGConv_3l_128h_w_k3, \
                  TAGConv_3l_512h_w_k3, GINNet, AttnGCN, TAGConv_9l_128h_k3,\
                  TAGConv_9l_512h_nw_k3

NNs = {
    "GINNet": GINNet,
    "AttnGCN": AttnGCN,
    "TAGConv_3l_128h_w_k3": TAGConv_3l_128h_w_k3,
    "TAGConv_3l_512h_w_k3": TAGConv_3l_512h_w_k3,
    "TAGConv_9l_128h_k3": TAGConv_9l_128h_k3,
    "TAGConv_9l_512h_nw_k3": TAGConv_9l_512h_nw_k3,
    
    
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
    for k in ["experiments", "simulator", "training"]:
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
    trainparams = dct["training"]
    
    expfiledict = {}   
    #iterate through datasets
    for datasetname in datasets:
        experimentlist = []
        #now iterate through experiments               
        expgen = combinationgenerator(experiments)
        datasetparams = datasets[datasetname]
            
        for expnum, experiment in enumerate(expgen):
            datafilename = os.path.join(workdir, projname+'_'+datasetname+'_exp'+str(expnum)+'.csv' )
            partfilename = os.path.join(workdir, projname+'_'+datasetname+'_exp'+str(expnum)+'.split' )
            
            updateddatasetparams = updateparams(datasetparams, experiment)
            simulateandsaveonedataset(updateddatasetparams, offset, datafilename, rng)
            
            savepartitions(datafilename, trainparams["valshare"], trainparams["testshare"], trainparams["partition_count"], partfilename, rng)
            
            expdict = { "id":expnum,
                       "experiment":experiment,
                       "datasetparams":datasetparams,
                       "datafile":datafilename,
                       "training": trainparams,
                       "partitionfile":partfilename
            }
            experimentlist.append(expdict)            
        expfiledict[datasetname] = experimentlist 
        
    experimentlistfile = os.path.join(workdir, outfile )
    #create file with all the experiments descritption
    with open(experimentlistfile,"w") as f:
        json.dump(expfiledict, f, indent=4, sort_keys=True)

def filterandsaveonedataset(indatafilename, outdatafilename, filters, cleanshare, cleandatafilename, rng):
    retval = None
    pairs, weights, labels, labeldict, idxtranslator = load_pure( indatafilename, debug=False, **(filters))
    conseq_pairs = translate_indices(pairs, idxtranslator)
    labellist = labeldict_to_labellist(labeldict)
    
    if not (cleanshare is None):
        #todo remove all edges to nodes from cleanshare into separate file
        graphdata = composegraphs(pairs, weights, labels, labeldict, idxtranslator)
        ncls = graphdata[0]['nodeclasses']
        grph = graphdata[0]['graph']
        trns = graphdata[0]['translation']
        #translate indices in ncls to original indices
        permt = getrandompermutation(ncls, rng)
        trainnodeclasses, valnodeclasses, testnodeclasses = dividetrainvaltest(ncls, 0, cleanshare, permt)
        part_ok, part_errors = checkpartition(grph, trainnodeclasses, valnodeclasses, testnodeclasses, details=False, trns=trns)
        if not part_ok:
            print("bad partition for clean dataset part")                    
        trainnodes, _, testnodes = gettrainvaltestnodes(trainnodeclasses, valnodeclasses, testnodeclasses)
        clnodes = idxtranslator[testnodes].tolist()
        #print("clean nodes:", clnodes)
        cllabels = [labellist[lbl] for lbl in labels[testnodes] ] 
        #print("clean node labels:", cllabels)
        retval = {"cleannodes":clnodes, "cleannodelabels":cllabels}
        with open(outdatafilename, 'w', encoding="utf-8") as f:
            with open(cleandatafilename, 'w', encoding="utf-8") as f2:
                f.write('node_id1,node_id2,label_id1,label_id2,ibd_sum,ibd_n\n')
                f2.write('node_id1,node_id2,label_id1,label_id2,ibd_sum,ibd_n\n')
                for idx, pair in enumerate(pairs):
                    i = conseq_pairs[idx][0]
                    j = conseq_pairs[idx][1]                    
                    label_i = labellist[labels[i]]
                    label_j = labellist[labels[j]]
                    name_i = label_i if "," not in label_i else '\"' + label_i + '\"'
                    name_j = label_j if "," not in label_j else '\"' + label_j + '\"'
                    if (i in trainnodes) and (j in trainnodes):                        
                        f.write(f'node_{pair[0]},node_{pair[1]},{name_i},{name_j},{weights[idx][0]},{pair[2]}\n')
                    else:
                        if i in testnodes:                            
                            if j in testnodes:
                                #we do not want edges between clean nodes
                                pass
                            else:
                                f2.write(f'node_{pair[0]},node_{pair[1]},unknown,{name_j},{weights[idx][0]},{pair[2]}\n')
                        elif j in testnodes:
                            f2.write(f'node_{pair[0]},node_{pair[1]},{name_i},unknown,{weights[idx][0]},{pair[2]}\n')
                        else:
                            print("ERROR! No node from pair belong to train or clean partitions")

        
    else:
        with open(outdatafilename, 'w', encoding="utf-8") as f:
            f.write('node_id1,node_id2,label_id1,label_id2,ibd_sum,ibd_n\n')
            for idx, pair in enumerate(pairs):
                i = conseq_pairs[idx][0]
                j = conseq_pairs[idx][1]
                label_i = labellist[labels[i]]
                label_j = labellist[labels[j]]
                name_i = label_i if "," not in label_i else '\"' + label_i + '\"'
                name_j = label_j if "," not in label_j else '\"' + label_j + '\"'
                f.write(f'node_{pair[0]},node_{pair[1]},{name_i},{name_j},{weights[idx][0]},{pair[2]}\n')
            
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
    trainparams = dct["training"]
    
    expfiledict = {}   
    #iterate through datasets
    for datasetname in datasets:
        print("processing dataset", datasetname)
        experimentlist = []        
        datasetparams = datasets[datasetname]
                
        indatafilename = os.path.join(datadir, datasetname+'.csv' )
        outdatafilename = os.path.join(workdir, projname+'_'+datasetname+'.csv' )
        partfilename = os.path.join(workdir, projname+'_'+datasetname+'.split' )
        
        cleanshare = None
        cleandatafilename = ""
        if "cleanshare" in trainparams:
            cleanshare = trainparams["cleanshare"]
            cleandatafilename = os.path.join(workdir, projname+'_'+datasetname+'_clean.csv' )
            
        
        retval = filterandsaveonedataset(indatafilename, outdatafilename, datasetparams["filters"], cleanshare, cleandatafilename, rng)

        savepartitions(outdatafilename, trainparams["valshare"], trainparams["testshare"], trainparams["partition_count"], partfilename, rng)

        expdict = {
                   "datafile":outdatafilename,
                   "training": trainparams,
                   "partitionfile":partfilename
        }
        if "cleanshare" in trainparams:
            expdict["cleanfile"] = cleandatafilename
            expdict["cleannodes"] =  retval["cleannodes"]
            expdict["cleannodelabels"] = retval["cleannodelabels"]
        experimentlist.append(expdict)            
        expfiledict[datasetname] = experimentlist 
        
    experimentlistfile = os.path.join(workdir, outfile )
    #create file with all the experiments descritption
    with open(experimentlistfile,"w", encoding="utf-8") as f:
        json.dump(expfiledict, f, indent=4, sort_keys=True)
            

def rungnn(workdir, infile, rng):
    with open(os.path.join(workdir, infile),"r") as f:
        explist = json.load(f)
    
    result = {}
    #todo: unique runname for every run
    #todo: multiple NNs
    
    for dataset in explist:
        print("Running experiments for", dataset)
        datasetexplist = explist[dataset]
        datasetresults = []
        for exp_idx, exp in enumerate(datasetexplist):
            
            datafile = os.path.join(workdir, exp["datafile"])
            
            with open(os.path.join(workdir, exp["partitionfile"]),"r") as f:
                partitions = json.load(f)
            expresults = {nnclass:[] for nnclass in NNs} 
            for part_idx, partition in enumerate(partitions["partitions"]):

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
                for nnclass in NNs:
                    run_name = os.path.join(workdir, "runs", "run_"+dataset+"_exp"+str(exp_idx)+"_split"+str(part_idx)+"_"+nnclass)
                    runresult = simplified_genlink_run(datafile, train_split, valid_split, test_split, run_name, NNs[nnclass]) 
                    print("RUN COMPLETE!", nnclass, runresult)
                    expresults[nnclass].append(runresult)
            
            print("experiment results for different splits:", expresults)
            datasetresults.append({nnclass: {"mean": np.average(expresults[nnclass]), 
                                             "std": np.std(expresults[nnclass]), 
                                             "values":expresults[nnclass]} for nnclass in NNs})
            #now clean test if requested
            if "cleanfile" in exp:
                print("Running clean inference test")
                #prepare dataframes once for all neuronetworks
                #compose df from dfmain and edges to node from dfclean
                cleanfile = os.path.join(workdir, exp["cleanfile"])
                dfmain = pd.read_csv(datafile)
                dfclean = pd.read_csv(cleanfile)
                    
                
                cleantestdataframes = {}
                cleannodes = exp["cleannodes"]
                #cleannodestxt = ["node_"+str(nd) for nd in cleannodes]
                cleannodelabels = exp["cleannodelabels"]
                for node in cleannodes:
                    fltr1 = dfclean[dfclean["node_id1"] =="node_" +str(node)]
                    #fltr1 = fltr1[not (fltr1["node_id2"] in cleannodestxt)]
                    fltr2 = dfclean[dfclean["node_id2"] =="node_" +str(node)]                    
                    #fltr2 = fltr2[not (fltr2["node_id1"] in cleannodestxt)]
                    onenodedf = pd.concat([dfmain, fltr1, fltr2])
                    #onenodedf.reset_index(drop=True)
                    cleantestdataframes[node] = onenodedf.reset_index(drop=True)
                
                ##               
                expresults = {nnclass:[] for nnclass in NNs}
                for part_idx, partition in enumerate(partitions["partitions"]):                    
                    for nnclass in NNs:
                        run_name = os.path.join(workdir, "runs", "run_"+dataset+"_exp"+str(exp_idx)+"_split"+str(part_idx)+"_"+nnclass, "model_best.bin" )
                        inferredlabels = []
                        for node in cleannodes:
                            print("infering class for node", node)
                            testresult = independent_test(run_name, NNs[nnclass], cleantestdataframes[node], node )
                            print("clean test classification", testresult)
                            inferredlabels.append( testresult )

                        runresult = f1_score(cleannodelabels, inferredlabels, average='macro')
                        expresults[nnclass].append(runresult)
                
                for nnclass in NNs:
                    datasetresults[exp_idx][nnclass].update({"clean_mean": np.average(expresults[nnclass]), 
                                             "clean_std": np.std(expresults[nnclass]), 
                                             "clean_values":expresults[nnclass]} )
            
        result[dataset] = datasetresults
    return result            
            
            
def runandsavegnn(workdir, infile, outfile, rng):
    result = rungnn(workdir, infile, rng)
    with open(os.path.join(workdir, outfile),"w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, sort_keys=True)            
            

            
def runandsaveall(workdir, infile, outfile, rng):
    result = rungnn(workdir, infile, rng)
    result2 = runheuristics(workdir, infile, rng)
    for dataset in result:        
        exps = result[dataset]
        exps2 = result2[dataset]
        
        for idx in range(len(exps)):
            exps[idx].update(exps2[idx])
    with open(os.path.join(workdir, outfile),"w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, sort_keys=True)            


#TODO runname -> workdir
def simplified_genlink_run(dataframe_path, train_split, valid_split, test_split, run_name, nnclass):
    '''
        returns f1 macro for one experiment
    '''
    dp = DataProcessor(dataframe_path)

    dp.load_train_valid_test_nodes(train_split, valid_split, test_split, 'numpy')

    dp.make_train_valid_test_datasets_with_numba('one_hot', 'homogeneous', 'multiple', 'multiple', run_name, log_edge_weights=False)

    trainer = Trainer(dp, nnclass, 0.0001, 5e-5, torch.nn.CrossEntropyLoss, 10, run_name, 2, 20,
                      'graph_based', 1, 
                      cuda_device_specified=1)

    def __init__(self, data: DataProcessor model_cls, lr, wd, loss_fn, batch_size, log_dir, patience, num_epochs, 
                 
                 feature_type, train_iterations_per_sample, evaluation_steps, weight=None, cuda_device_specified: int = None)
    
    
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