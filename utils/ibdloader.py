import pandas
import numpy as np
import networkx as nx

def stripnodename(df):
    '''inplace change node_id fields into int id fields    
    '''
    df['node_id1'] = df['node_id1'].apply(lambda x:int(x[5:]))
    df['node_id2'] = df['node_id2'].apply(lambda x:int(x[5:]))
    
    
    
def removeweakclasses(df, weaklabels):
    for c in weaklabels:
        print("dropping", c)
        df.drop(df[df['label_id1']==c].index, inplace=True)
        df.drop(df[df['label_id2']==c].index, inplace=True)
        

def getuniqnodes(df, dftype):
    df1 =  \
    df[['node_id1','label_id1']].rename(columns={'node_id1':'node_id','label_id1':'label_id'})
    df2 =  \
    df[['node_id2','label_id2']].rename(columns={'node_id2':'node_id','label_id2':'label_id'})
    uniqnodes = pandas.concat([df1,df2]).drop_duplicates()
    nodecount = uniqnodes.shape[0]
    uniqids = uniqnodes.drop_duplicates('node_id')
    idcount = uniqids.shape[0]
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

    
def load_metis(weights_labels_file, ibd_labels_file):
    '''Verify and load files from Genotek 2-population metis format
        into numpy arrays
    
    Parameters
    ----------
    weights_labels_file: str
        filename for the list of edges with weights and node descriptions         
    ibd_labels_file: str
        filename for the list of edges with node descriptions
    
    Returns
    -------
    simplepairs: ndarray 
        array of N x 3 ints for unique edges from ibd_labels_file
        each row is of the form [node1, node2, number of occurences]
    weightedpairs: ndarray
        array of M x 2 ints for edges from weights_labels_file
    weights: ndarray
        array of M floats of weights from weights_labels_file
        k-th weight corresponds to the k-th row of the weightedpairs
    CRrate: ndarray
        array of floats from [0,1] with a fraction of Central Russian data in genome        
    '''
    
    print("Loading ibd data files")
    dfweights = pandas.read_csv(weights_labels_file)
    dfpairs = pandas.read_csv(ibd_labels_file)
    stripnodename(dfweights)
    stripnodename(dfpairs)
    
    #get unique nodes
    uniqidsw = getuniqnodes(dfweights, 'weights')
    uniqidsp = getuniqnodes(dfpairs, 'pairs')

    uniqids = pandas.concat([uniqidsw,uniqidsp]).drop_duplicates()
    idcount = uniqids.shape[0]
    print("Unique ids total:", idcount)
    if (idcount != uniqidsp.shape[0]) or (idcount != uniqidsw.shape[0]):
        print("WARNING: ibd ans weights labels inconsistent!")
        

    #create np arrays for fast access:
    #[CR_rate] for node origin data
    #[[id1,id2]] and [ibd weight] with corresponding index for weighted ibd
    #[[id1,id2]] for unweighted ibd

    #check if ids are consecutive
    checkuniqids(uniqids)
    
    CRrate = np.zeros((idcount),dtype=float)
    for index, row in uniqids.iterrows():
        #print(row)
        id = row['node_id']
        ratedictlist = eval(row['label_id'])
        ratedict={}
        for item in ratedictlist:
            ratedict.update(item)
        cr = 0.0
        if 'Central-Russia' in ratedict:
            cr = float(ratedict['Central-Russia'])/100.0
            if 'Jews' in ratedict: 
                jw = float(ratedict['Jews'])/100.0
                if cr+jw<1.0-0.00001:
                    print ("for node id=", id, "rates sum to", cr+jw)
            else:
                if cr<1.0-0.00001:
                    print ("for node id=", id, "only cr rate is", cr)   
        else:
            if 'Jews' in ratedict:
                jw = float(ratedict['Jews'])/100.0
                if jw<1.0-0.00001:
                    print ("for node id=", id, "only jw rate is", jw)
            else:
                print("Empty dict!")
        CRrate[id] = cr
    print("CRrate filling complete")
    print("Number of 100% CR:",np.sum(CRrate>1.0-0.00001))
    print("Number of 100% Jews:",np.sum(CRrate<0.00001))
    #print (CRrate[10900:])
    #print (CRrate[291])
    weightedpairscount = dfweights.shape[0]
    simplepairscount = dfpairs.shape[0]
    print("Checking duplicates:")
    print("  weghted:", weightedpairscount, "->", dfweights.drop_duplicates().shape[0])
    print("  simple:", simplepairscount, "->", dfpairs.drop_duplicates().shape[0])
    #did not check reversed duplicates
    weightedpairs = dfweights[['node_id1', 'node_id2']].to_numpy()
    weights = dfweights['ibd_sum'].to_numpy()

    dfnodepairs = dfpairs[['node_id1', 'node_id2']]
    #these are with repetitions, let's count them
    simplepairs = dfnodepairs.groupby(dfnodepairs.columns.tolist(),as_index=False).size().to_numpy()
    #print(dfnodepairs)
    #simplepairs = dfnodepairs.to_numpy()
    #print(simplepairs[-100:])
    print("Simple pairs shape:",simplepairs.shape, ", total occurences:", np.sum(simplepairs[:,2]))

    #print(weightedpairs[:10])
    #print(weights[:10])
    if (np.sum(simplepairs[:,0]>=simplepairs[:,1]) > 0):
        print("WARNING! Some pairs are unordered, repetition can be missed")
    else:
        print("OK: pairs are ordered")

    print("Finished converting dataframes to numpy") 
    
    return simplepairs, weightedpairs, weights, CRrate



def load_pure(datafilename, minclassize=None, removeclasses=None):
    '''Verify and load files from Genotek dataset1 pure format
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
        removeweakclasses(dfibd, removeclasses)   
    uniqids = getuniqnodes(dfibd, 'ibd')
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
        print("Removing following classes:")
        totalweak = 0
        for idx in range(len(weakargs)):        
            sids = weakargs[idx]
            totalweak += weaklabelcount[sids]
            print(weaklabels[sids], weaklabelcount[sids])
        print("Total", totalweak, "removed")
        print("Remaining classes:")
        for idx in range(len(powerargs)):        
            sids = powerargs[idx]
            print(powerlabels[sids], powerlabelcount[sids])

        #remove rare from uniqids
        removeweakclasses(dfibd, weaklabels)

        uniqids = getuniqnodes(dfibd, 'ibd')
        checkuniqids(uniqids)
        uniqids = uniqids.sort_values(by=['node_id'])

        labeldf = uniqids[['label_id']].drop_duplicates()

        #compile label dictionary
        lbl = 0    
        labeldict = {}
        for idx in powerargs[::-1]:            
            labeldict[powerlabels[idx]] = lbl
            lbl += 1        
    
    print("Label dictionary:", labeldict)
    #create labels array
    labels = uniqids['label_id'].apply(lambda x:labeldict[x]).to_numpy()    
    idxtranslator = uniqids['node_id'].to_numpy()    
    #id translator[idx] contains idx's node name
    #it equals to idx if nodes are named from 0 consistently
    
    dfnodepairs = dfibd[['node_id1', 'node_id2']]
    paircount = dfnodepairs.shape[0]
    uniqpaircount = dfnodepairs.drop_duplicates().shape[0]
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