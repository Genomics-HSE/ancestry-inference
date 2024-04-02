import pandas
import numpy as np
import networkx as nx

def stripnodename(df):
    '''inplace change node_id fields into int id fields    
    '''
    df['node_id1'] = df['node_id1'].apply(lambda x:int(x[5:]))
    df['node_id2'] = df['node_id2'].apply(lambda x:int(x[5:]))
    
    
    
def removeweakclasses(df, weaklabels, debug=True):
    for c in weaklabels:
        if debug:
            print("dropping", c)
        df.drop(df[df['label_id1']==c].index, inplace=True)
        df.drop(df[df['label_id2']==c].index, inplace=True)
        

def getuniqnodes(df, dftype, debug=True):
    df1 =  \
    df[['node_id1','label_id1']].rename(columns={'node_id1':'node_id','label_id1':'label_id'})
    df2 =  \
    df[['node_id2','label_id2']].rename(columns={'node_id2':'node_id','label_id2':'label_id'})
    uniqnodes = pandas.concat([df1,df2]).drop_duplicates()
    nodecount = uniqnodes.shape[0]
    uniqids = uniqnodes.drop_duplicates('node_id')
    idcount = uniqids.shape[0]
    if debug:
        if nodecount!=idcount:
            print(f"WARNING! inconsistent labels in {dftype} datafile")
        print(f"Unique ids in {dftype} datafile: {idcount}")
    return uniqids
    
def checkuniqids(uniqids):
    idints = uniqids['node_id']
    if min(idints)>0:
        print("WARNING: ids are starting from", min(idints), "must be translated!")
    else:
        print("OK: Ids are starting from 0")
    if max(idints)-min(idints)+1 == idints.shape[0]:
        print("OK: Ids are consecutive")
    else:
        print("WARNING: ids are not consequtive, must be translated!")

def load_pure(datafilename, minclassize=None, removeclasses=None, only=None, debug=True):
    '''Verify and load files from dataset1 pure format
        into numpy arrays
    
    Parameters
    ----------
    datafilename: str
        filename for the list of edges with weights, counts and node descriptions         
    minclassize: int
        minimum class size to be included to returned nodes 
    removeclasses: list
        list of labels to remove from dataset
    Returns
    -------
    pairs: ndarray 
        array of N x 3 ints for ibd counts
        each row is of the form [node1, node2, number of ibd]
    weights: ndarray
        array of M floats of weights 
        k-th weight corresponds to the k-th row of the pairs
        if ibd_max column is present, then also contains a column for max values
    labels: ndarray
        array of ints with class number for every node
    labeldict: dict of labels
    idxtranslator: i-th node_id in the i-th element
    '''
    dfibd = pandas.read_csv(datafilename)
    stripnodename(dfibd)
    if not(removeclasses is None):
        removeweakclasses(dfibd, removeclasses, debug)   
    uniqids = getuniqnodes(dfibd, 'ibd', debug)
    if debug:
        checkuniqids(uniqids)
    uniqids = uniqids.sort_values(by=['node_id'])
    
    labeldf = uniqids[['label_id']].drop_duplicates()
        
    #compile label dictionary
    lbl = 0    
    labeldict = {}
    for _, row in labeldf.iterrows():
        labeldict[row['label_id']] = lbl
        lbl += 1        
    
    if not (minclassize is None):
        if debug:
            print("Filter out all classes smaller than ", minclassize)
        #count and filter out rare label_id
        powerlabels = []
        powerlabelcount = []
        weaklabels = []
        weaklabelcount = []
        for c in labeldict:        
            count = len(uniqids[uniqids['label_id']==c])
            if count<minclassize:
                weaklabels.append(c)
                weaklabelcount.append(count)
            else:
                powerlabels.append(c)
                powerlabelcount.append(count)
        weakargs = np.argsort(weaklabelcount)
        powerargs = np.argsort(powerlabelcount)
        if debug:
            print("Removing following classes:")
        totalweak = 0
        for idx in range(len(weakargs)):        
            sids = weakargs[idx]
            totalweak += weaklabelcount[sids]
            if debug:
                print(weaklabels[sids], weaklabelcount[sids])
        if debug:
            print("Total", totalweak, "removed")
            print("Remaining classes:")
        for idx in range(len(powerargs)):        
            sids = powerargs[idx]
            if debug:
                print(powerlabels[sids], powerlabelcount[sids])

        #remove rare from uniqids
        removeweakclasses(dfibd, weaklabels, debug)

        uniqids = getuniqnodes(dfibd, 'ibd', debug)
        if debug:
            checkuniqids(uniqids)
        uniqids = uniqids.sort_values(by=['node_id'])

        labeldf = uniqids[['label_id']].drop_duplicates()

        #compile label dictionary
        lbl = 0    
        labeldict = {}
        for idx in powerargs[::-1]:            
            labeldict[powerlabels[idx]] = lbl
            lbl += 1        
    
    if debug:
        print("Label dictionary:", labeldict)
    #create labels array
    labels = uniqids['label_id'].apply(lambda x:labeldict[x]).to_numpy()    
    idxtranslator = uniqids['node_id'].to_numpy()    
    #id translator[idx] contains idx's node name
    #it equals to idx if nodes are named from 0 consistently
    
    dfnodepairs = dfibd[['node_id1', 'node_id2']]
    paircount = dfnodepairs.shape[0]
    uniqpaircount = dfnodepairs.drop_duplicates().shape[0]
    if debug:
        print(f"paircount: {paircount}, unique: {uniqpaircount}")
        if paircount!=uniqpaircount:
            print("Not OK: duplicate pairs")
        else:
            print("OK: all pairs are unique")
    
    pairs = dfibd[['node_id1', 'node_id2', 'ibd_n']].to_numpy()
    #todo order pairs
    for idx in range(pairs.shape[0]):
        n1 = pairs[idx,0]
        n2 = pairs[idx,1]
        if n1>n2:
            pairs[idx,0] = n2
            pairs[idx,1] = n1
        
    if 'ibd_max' in dfibd.columns:
        weights = dfibd[['ibd_sum', 'ibd_max']].to_numpy()
    else:
        weights = dfibd[['ibd_sum']].to_numpy()
    
    return pairs, weights, labels, labeldict, idxtranslator
     
    
def translate_indices(pairs, idxtranslator):
    """
        replaces nonconsecutive indices from pairs to consecutive according to idxtranslator
    
    """
    result = np.empty_like(pairs)
    for idx in range(pairs.shape[0]):
        result[idx,0] = np.where(idxtranslator==pairs[idx,0])[0][0]
        result[idx,1] = np.where(idxtranslator==pairs[idx,1])[0][0]
        result[idx,2] = pairs[idx,2]
    return result
    
    

def getcloserelatives_w(weightedpairs, weights, threshold):
    widelinks = weights>threshold
    print("number of wide links:", np.sum(widelinks))
    relatives = []
    for pair in weightedpairs[widelinks]:
        relatives.append(pair[0])
        relatives.append(pair[1])
    relatives = list(set(relatives))
    print("total close relatives (unique):", len(relatives))
    return relatives
    
def fill_bins(rates, binbounds):
    '''
        returns bins for cr rates according to bin bounds
    '''
    bins = [ [] for _ in range(len(binbounds)-1) ]
    for nodeidx, rate in enumerate(rates):
        for idx in range(len(bins)):
            if (binbounds[idx] <= rate) and (rate < binbounds[idx+1]):
                bins[idx].append(nodeidx)
    return bins

def get_bin_idx(rate, binbounds):
    '''
        returns bin index for specified cr rate
    '''
    for idx in range(len(binbounds)-1):
        if (binbounds[idx] <= rate) and (rate < binbounds[idx+1]):
            return idx
    return -1
    
    
def get_stat_for_node_vs_group(node, bn, Gw):
    '''
        returns number of links between node and a group in graph Gw
    '''
    weights = []
    for edge in nx.edge_boundary(Gw, bn, [node], data=True):
        _, _, d = edge
        weights.append(d["weight"])
    if len(weights)>0:
        weights = np.array(weights)
        return weights.shape[0], np.mean(weights), np.std(weights) 
    else:
        return 0,0,0

def get_weighted_dataloaders():
    pass


if __name__=='__main__':    
    simplepairs, weightedpairs, weights, CRrate = \
        load("CR_Jews_graph_weights_labels.csv","CR_Jews_graph_ibd_labels.csv")
    print(simplepairs[:10])
    am = np.argmax(simplepairs[:,2])
    print('edge with maximum number of occurences', simplepairs[am])
    
    print(weightedpairs[:10])    
    print(weights[:10])
    print(CRrate[:20])