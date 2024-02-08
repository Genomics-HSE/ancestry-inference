import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

import ibdloader


Gw = None
def init_weighted_graph(weightedpairs, weights, whitelist, blacklist = []):
    #blacklist вообще не добавлять ребро в граф, если хотя бы одна вершина из черного списка
    #whitelist добавлять ребро, если хотя бы одна из вершин в белом списке
    #
    global Gw
    Gw = nx.Graph()
    edgecount = 0
    for idx, edge in enumerate(weightedpairs):
        if (not (edge[0] in blacklist)) and (not (edge[1] in blacklist)):
            #if (edge[0] in purenodes) or (edge[1] in purenodes):
            Gw.add_edge(edge[0], edge[1], weight=weights[idx])
            edgecount +=1
            #print("added ", edgecount)
    print("Edges added:", edgecount)

def getnumberofedgesfromnode(node):
    return Gw.degree(node)

def getremainingedgecountpernode(nodes):
    result = [getnumberofedgesfromnode(node) for node in nodes]
    return result
        
def statspooladapter(arg):
    node, bn = arg
    count, wt_mean, wt_stdev = ibdloader.get_stat_for_node_vs_group(node, bn, Gw)
    return (count, wt_mean, wt_stdev)


def breaktraintest(bins, blacklist, trainshare, targetnames, permutate=False):
    # break bins into train and test parts, also returns filtered bins
    goodbins = []
    for bn in bins:
        goodbn = []
        for node in bn:
            if not node in blacklist:
                goodbn.append(node)
        goodbins.append(goodbn)
    #can not pickle lambda
    #goodbins = [filter((lambda x: x not in blacklist), bn ) for bn in bins]
    
    if permutate:
        rndsortedbins = [np.random.permutation(bn) for bn in goodbins]
    else:
        rndsortedbins = [bn for bn in goodbins]

    breakidx = [int(len(bn)*trainshare) for bn in goodbins]
    bins_train = [goodbins[idx][:breakidx[idx]] for idx in range(len(goodbins))]
    bins_test = [goodbins[idx][breakidx[idx]:] for idx in range(len(goodbins))]
    train_nodes = []
    for bn in bins_train:
        train_nodes = train_nodes + bn
    test_nodes = []
    for bn in bins_test:
        test_nodes = test_nodes + bn
    #print table
    print("{}  \t {} \t {} \t {} ".format("CR rate", "total", "train", "test") )
    for idx in range(len(bins)):
        print("{}%      \t {} \t {} \t {} ".format(targetnames[idx], len(bins[idx]), len(bins_train[idx]), len(bins_test[idx]) ) )
    
    binscount = len(bins)
    total_train = 0
    total_test = 0
    total_sum = 0
    for idx in range(binscount):
        total_train += len(bins_train[idx])
        total_test += len(bins_test[idx])
        total_sum += len(goodbins[idx])

    print("{}  \t {} \t {} \t {} ".format("Total:", total_sum, total_train, total_test) ) 
    return goodbins, bins_train, bins_test
    
    


def get_datasets_with_3_features(weightedpairs, weights, CRrate, binbounds, bins, bins_train, bins_test,
                                  keeplinkstobins, targetnames):
    
    #5. collect number of links to every bin and mean&stdev of their weigth
    from multiprocessing import Pool
    process_count = 48

    binscount = len(bins)
    total_train = 0
    total_test = 0
    for idx in range(binscount):
        total_train += len(bins_train[idx])
        total_test += len(bins_test[idx])
    train_nodes = []
    for bn in bins_train:
        train_nodes = train_nodes + bn
    test_nodes = []
    for bn in bins_test:
        test_nodes = test_nodes + bn
    
    featuredbinsidxs = []
    for idx in range(len(keeplinkstobins)):
        if keeplinkstobins[idx]:
            featuredbinsidxs.append(idx)
    featuredbinscount = sum(keeplinkstobins)
        
    x_train = np.zeros((total_train,3*featuredbinscount), dtype = np.float32)
    y_train = np.zeros((total_train), dtype = np.float32)
    x_test = np.zeros((total_test,3*featuredbinscount), dtype = np.float32)
    y_test = np.zeros((total_test), dtype = np.float32)

    print("Preparing train dataset with", total_train, "items")
    adapterargs = []
    for idx in range(total_train):
        node = train_nodes[idx] #next train node
        y_train[idx] = ibdloader.get_bin_idx(CRrate[node],binbounds) #get class
        for bnidx in featuredbinsidxs:
            adapterargs.append((node, bins[bnidx]))
    print("prepared ", len(adapterargs), "arguments for multiprocessing")

    with Pool(process_count) as p:
        adapterresults = list( tqdm(p.imap(statspooladapter, adapterargs) , total=total_train*featuredbinscount) )

    for idx in range(total_train):
        for fbi in range(len(featuredbinsidxs)):
            count, wt_mean, wt_stdev = adapterresults[idx*featuredbinscount + fbi] 
            x_train[idx][fbi*3+0] = count
            x_train[idx][fbi*3+1] = wt_mean
            x_train[idx][fbi*3+2] = wt_stdev
        
    print("Preparing test dataset with", total_test, "items")        

    adapterargs = []
    for idx in range(total_test):
        node = test_nodes[idx] 
        y_test[idx] = ibdloader.get_bin_idx(CRrate[node],binbounds)
        for bnidx  in featuredbinsidxs:
            adapterargs.append((node, bins[bnidx]))

    print("prepared ", len(adapterargs), "arguments for multiprocessing")

    with Pool(process_count) as p:
        adapterresults = list( tqdm(p.imap(statspooladapter, adapterargs) , total=total_test*featuredbinscount) )

    for idx in range(total_test):
        for fbi in range(len(featuredbinsidxs)):
            count, wt_mean, wt_stdev = adapterresults[idx*featuredbinscount + fbi] 
            x_test[idx][fbi*3+0] = count
            x_test[idx][fbi*3+1] = wt_mean
            x_test[idx][fbi*3+2] = wt_stdev
            
    return x_train, y_train, x_test, y_test