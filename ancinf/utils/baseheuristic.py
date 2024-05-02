import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
import scipy
import scipy.stats


def plot_distribution(data, threshold, title):
    print(title)
    pdata = data[data<threshold]
    print ("Total values:",len(data), "min:", min(data), "max:",max(data), "ave:", np.average(data))
    print ("filtered values:", len(pdata), "min:", min(pdata), "max:", max(pdata), "ave:", np.average(pdata))
    plt.rcParams["figure.figsize"] = (24,9)
    plt.hist(pdata, bins = 'auto')#, width=barWidth)
    plt.title(title)
    plt.show()
    
def plot_and_fit_distribution(data, threshold, bins, title, dist_names):
    print(title)
    pdata = data[data<threshold]
    print ("Total values:",len(data), "min:", min(data), "max:",max(data), "ave:", np.average(data))
    print ("filtered values:", len(pdata), "min:", min(pdata), "max:", max(pdata), "ave:", np.average(pdata))
    plt.rcParams["figure.figsize"] = (24,9)
        
    x = np.arange(threshold)
    h = plt.hist(pdata, bins = bins)
    frequencies,bns,_ = h
    binsize = bns[1]-bns[0]
    bincenters = bns[:-1] + binsize/2.
    size = binsize*pdata.shape[0]
    
    for dist_name in dist_names:
        dist = getattr(scipy.stats, dist_name)
        params = dist.fit(pdata)
        pdf_fitted = dist.pdf(x, *params) * size        
        pdf_values = dist.pdf(bincenters, *params) * size     
        sse = np.sum(np.power( (frequencies - pdf_values), 2.0))    
        plt.plot(pdf_fitted, label=dist_name+":"+str(sse))
        plt.scatter(bincenters, pdf_values )
        plt.xlim(0.1,threshold)
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()
    

    
def show_prediction(labels, prediction, display_labels, exp_title):   
    acc_string = f" Accuracy: {np.sum(labels == prediction)/labels.shape[0]:.4f}, correct: {np.sum(labels == prediction)}, total: {labels.shape[0]}"
    print(exp_title+":"+acc_string)
    print(classification_report(labels, prediction, target_names = display_labels))
    cm = confusion_matrix(labels, prediction)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=display_labels).plot()        
    cm_display.ax_.set_title(exp_title+":"+acc_string)
    plt.show()
    #cm_display.show()
    
    
    
def getIntBoundary(G, group):
    '''
        for every node in the group we collect ibd sum to the rest of the group
        also we return number of edges where evey edge is counted twice
    '''
    distr = []
    edgecount = 0
    for node in group:
        sumweight = 0
        for edge in nx.edge_boundary(G, [node], group, data=True):
            _, _, d = edge
            sumweight += d["weight"][0]
            edgecount +=1
        distr.append(sumweight)          
    return distr, edgecount

def getExtBoundary(G, group1, group2):
    '''
        for every node in the group1 we collect ibd sum to the group 2
        also we return number of edges between group1 and group2
    '''
    distr = []
    edgecount = 0
    for node in group1:
        sumweight = 0
        for edge in nx.edge_boundary(G, [node], group2, data=True):
            _, _, d = edge
            sumweight += d["weight"][0]
            edgecount +=1
        distr.append(sumweight)          
    return distr, edgecount

def getweightandcountmatrices(G, nodeclasses, labeldict, plot_distr = False):
    '''
         compute edge distributon parameters:
         total weigth and
         total count
         
    Parameters
    ----------
    G: nx graph
        input graph
    nodeclasses: dict
        population:list of nodes
    labeldict: dictionary
        contains int index of every text label
    plot_distr: bool
        interactive output with plots
    
    Returns
    -------
    weimatrix: 2d np array
        sum of edge weights
    countmatrix: 2d np array
        number of edges
    distributions: dict
        label or pair of labels:edge weights
    '''
    
    labelcount = len(labeldict)
    weimatrix = np.zeros((labelcount,labelcount))
    countmatrix = np.zeros((labelcount,labelcount))
    distributions = {}

    for label in labeldict:
        #print(label)
        intdistr, ec = getIntBoundary(G, nodeclasses[label] )   
        if plot_distr:
            plot_distribution(np.array(intdistr), 1000, f"Distribution of ibd sum for {label}")
            print("------------------------------------------------------------------------")
        
        
        weimatrix[labeldict[label], labeldict[label] ] = np.sum(intdistr)
        
        countmatrix[labeldict[label], labeldict[label] ] = ec /2 #in ec every edge is counted twice
        
        distributions[label] = {"data":np.array(intdistr), "threshold":None, "bins":'auto'}

    for baselabel in labeldict:
        for destlabel in labeldict:
            if destlabel!= baselabel:
                #print(f"from {baselabel} to {destlabel}")
                intdistr, ec = getExtBoundary(G, nodeclasses[baselabel], nodeclasses[destlabel] )
                weimatrix[labeldict[destlabel], labeldict[baselabel] ] = np.sum(intdistr)
                countmatrix[labeldict[destlabel], labeldict[baselabel] ] = ec
                distributions[baselabel+"_to_"+destlabel] = {"data":np.array(intdistr), "threshold":None, "bins":'auto'}
                if plot_distr:
                    plot_distribution(np.array(intdistr), 1000, f"Distribution of ibd sum for {baselabel} to {destlabel} ")
                    print("------------------------------------------------------------------------")   
    return weimatrix, countmatrix, distributions

def getprobandmeanmatrices(G, nodeclasses, labeldict):
    '''
         compute edge distributon parameters:
         mean ibdsum on the edge between classes
         and probability of the edge
         
    Parameters
    ----------
    G: nx graph 
        input graph
    nodeclasses: dict
        {population_i:[list of nodes]}
    labeldict: dictionary
        contains int index of every text label
    
    Returns
    ----------
    weimatrix: 2d np array 
        sum of edge weights divided by number of edges (mean edge weight)
    countmatrix: 2d np array
        number of edges divided by total possible edges (probability of the edge)
    '''
    
    labelcount = len(labeldict)
    weimatrix = np.zeros((labelcount,labelcount)) #mean weight
    countmatrix = np.zeros((labelcount,labelcount)) #edge probability
   
    for label in labeldict:
        intdistr, ec = getIntBoundary(G, nodeclasses[label] )
        if ec>0:
            weimatrix[labeldict[label], labeldict[label] ] = np.sum(intdistr)/ec
            maxpossibleedges = len(nodeclasses[label])*(len(nodeclasses[label])-1) / 2            
            countmatrix[labeldict[label], labeldict[label] ] = ec / 2 / maxpossibleedges             
        else:
            weimatrix[labeldict[label], labeldict[label] ] = -1
            countmatrix[labeldict[label], labeldict[label] ] = 0
            

    for baselabel in labeldict:
        for destlabel in labeldict:
            if destlabel!= baselabel:
                intdistr, ec = getExtBoundary(G, nodeclasses[baselabel], nodeclasses[destlabel])
                if ec>0:
                    weimatrix[labeldict[destlabel], labeldict[baselabel] ] = np.sum(intdistr)/ec
                    maxpossibleedges = len(nodeclasses[baselabel])*len(nodeclasses[destlabel])
                    countmatrix[labeldict[destlabel], labeldict[baselabel] ] = ec/maxpossibleedges
                else:
                    weimatrix[labeldict[destlabel], labeldict[baselabel] ] = -1
                    countmatrix[labeldict[destlabel], labeldict[baselabel] ] = 0
                
    return weimatrix, countmatrix



def translate_and_filter(pairs, weigths, translation):
    """
        replaces nonconsecutive indices from pairs to consecutive according to translation
        and removes pairs and weights with nodes not occuring in translation
    
    """
    outpairs = []
    outweights = []
    for idx in range(pairs.shape[0]):
        if (pairs[idx,0] in translation) and (pairs[idx,1] in translation):
            outweights.append(weigths[idx])
            outpairs.append([
                            np.where(translation==pairs[idx,0])[0][0],
                            np.where(translation==pairs[idx,1])[0][0],
                            pairs[idx,2] ] )
    return np.array(outpairs), np.array(outweights)


def composegraphs(pairs, weights, labels, labeldict, translation, train=None, test=None):
    '''
        Create graph(s) with edges given by pairs for nodes specified by train and test
        If no train nodes provided, returns one graph with every node in translation
        If train nodes are provided, but test nodes are not, returns one graph restricted to train nodes
        If both train and test are provided, returns a graph for every test node, containg all train and this test node
    
    Parameters
    ----------
    pairs: np array of [node_id1 node_id2 ibd_count] 
        graph edges
    weights: np array of float
        weight of every edge
    labels: np array of int
        labels of 
    labeldict: dictionary
        contains int index of every text label
    translation: np array of int
        label[idx] is the label of translation[idx] node from dataset
    train and test: arrays of int 
        disjoint subsets of range(len(translation))
    
        
    Returns
    -------
    result: list of dictionaries
        graph: resuting graph with nodes starting from 0 and consecutive
        translation: i-th element is dataset node code of the node i in the graph
        labels: i-th element is label of the node i in the graph
        nodeclasses: dictionary with graph nodes (starting from 0 and consequtive) grouped by label
    '''
    if train is None:
        #only full graph
        goodpairs, goodweights = translate_and_filter(pairs, weights, translation)
        G = nx.Graph()
        for idx, edge in enumerate(goodpairs):
            G.add_edge(edge[0], edge[1], weight=goodweights[idx])
        
        #print (labels.shape)
        nodeclasses = {}
        for label in labeldict:
            idx = labeldict[label]
            nodeclasses[label] = np.argwhere(labels==idx).flatten()    
        return [{"graph":G, "labels": labels, "translation": translation, "nodeclasses": nodeclasses}]
    else:
        if test is None:
            #only one graph with selected(train) nodes
            G = nx.Graph()
            goodpairs, goodweights = translate_and_filter(pairs, weights, translation[train])
                
            for idx, edge in enumerate(goodpairs):
                G.add_edge(edge[0], edge[1], weight=goodweights[idx])
            
            
            filteredlabels = labels[train]
            
            nodeclasses = {}
            
            for label in labeldict:
                idx = labeldict[label]
                nodeclasses[label] = np.argwhere(filteredlabels==idx).flatten()
            return [{"graph": G, "labels": filteredlabels, "translation": translation[train], "nodeclasses": nodeclasses}]
        else:
            result = []
            for testnode in test:
                G = nx.Graph()                
                trainplusonetest = train + [testnode]
                #print ("tpp", trainplusonetest)
                goodpairs, goodweights = translate_and_filter(pairs, weights, translation[trainplusonetest])
                
                for idx, edge in enumerate(goodpairs):
                    G.add_edge(edge[0], edge[1], weight=goodweights[idx])
                
                filteredlabels =labels[trainplusonetest]
                nodeclasses = {}
                for label in labeldict:
                    idx = labeldict[label]
                    nodeclasses[label] = np.argwhere(filteredlabels==idx).flatten()

                result.append({"graph": G, "labels": filteredlabels, "translation": translation[trainplusonetest], "nodeclasses": nodeclasses})
            return result




def getlinkmult(node1, node2, pairs):
    if node1<node2:
        n1, n2 = node1, node2
    else:
        n1, n2 = node2, node1
    fltr = np.logical_and((pairs[:,0]==n1), (pairs[:,1]==n2))
    return pairs[fltr][0][2]




def getfeatures(G, testnodes, nodeclasses, labeldict,  pairs, translation ):
    '''Collect features of graph nodes
    
    Parameters
    ----------
    G: networkx graph
        graph to process. Nodes of G are consequtive integers starting from 0
    testnodes: np array of int
        requested subset of nodes of G to collect features for
    nodeclasses: np array of int
        nodesclasses[i] is the class of i-th node
    labeldict: dictionary
        {'class0':0, 'class101':1 etc}
    pairs: np array of int
        first two columns are node indices forming an edge, third row is the number of ibd segments on this edge
    translation:  
        correspondence between graph node names and node names in pairs
        
    Returns
    -------
    result: dictionary of features
        each feature has 'data' field which size equals to graph node count
        features for requested nodes are filled, others are zeroed
    '''
    nodecount = len(G.nodes)
    labelcount = len(labeldict)

    result = {}
    result.update({"EdgeCount": {"data": np.zeros(nodecount*labelcount, dtype = np.float32).reshape((nodecount, labelcount)),
                                   "comment":"number of edges from node to population"} })
    result.update({"EdgeCountPerClassize": {"data": np.zeros(nodecount*labelcount, dtype = np.float32).reshape((nodecount, labelcount)),
                                              "comment":"number of edges from node to population per population size"} })
    result.update({"SegmentCount": {"data": np.zeros(nodecount*labelcount, dtype = np.float32).reshape((nodecount, labelcount)),
                                        "comment":"number of shared ibd segments between node and population"} })
    result.update({"LongestIbd": {"data": np.zeros(nodecount*labelcount, dtype = np.float32).reshape((nodecount, labelcount)),
                                 "comment":"longest shared ibd segment between node and population"} })
    result.update({"IbdSum": {"data": np.zeros(nodecount*labelcount, dtype = np.float32).reshape((nodecount, labelcount)),
                             "comment":"sum of shared ibd segment lengths between node and population"} })
    result.update({"IbdSumPerEdge": {"data": np.zeros(nodecount*labelcount, dtype = np.float32).reshape((nodecount, labelcount)),
                                    "comment":"sum of shared ibd segment lengths between node and population per edge count"} })

    for nd in testnodes:
        #get links to every class
        node_weights_to_classes = np.zeros(labelcount, dtype = np.float32)
        node_edges_to_classes = np.zeros(labelcount, dtype = np.int64)
        node_edges_to_classes_wmult = np.zeros(labelcount, dtype = np.int64)
        node_edges_to_classes_per_classize = np.zeros(labelcount, dtype = np.float32)
        node_longest_ibd = np.zeros(labelcount, dtype = np.float32)
        node_weight_per_segment = np.zeros(labelcount, dtype = np.float32)
        for label in labeldict:
            labelidx = labeldict[label]
            destination_class = nodeclasses[label]
            sumweight = 0
            edgecount = 0
            edgecountwmult = 0
            longest_ibd = -1

            for edge in nx.edge_boundary(G, [nd], destination_class, data=True):
                n1, n2, d = edge
                orig_n1 = translation[n1]
                orig_n2 = translation[n2]
                sumweight += d["weight"][0]
                edgecount +=1
                edgecountwmult += getlinkmult(orig_n1,orig_n2, pairs)
                if d["weight"][0]>longest_ibd:
                    longest_ibd = d["weight"][0]

            node_weights_to_classes[labelidx] = sumweight
            node_edges_to_classes[labelidx] = edgecount
            node_edges_to_classes_wmult[labelidx] = edgecountwmult
            node_edges_to_classes_per_classize[labelidx] = edgecount/len(destination_class)
            node_longest_ibd[labelidx] = longest_ibd
            if edgecount>0:
                node_weight_per_segment[labelidx] = sumweight / edgecount
            else:
                edgecount = -1


        result["EdgeCount"]["data"][nd] = node_edges_to_classes
        result["EdgeCountPerClassize"]["data"][nd] = node_edges_to_classes_per_classize
        result["SegmentCount"]["data"][nd] = node_edges_to_classes_wmult
        result["LongestIbd"]["data"][nd] = node_longest_ibd
        result["IbdSum"]["data"][nd] = node_weights_to_classes
        result["IbdSumPerEdge"]["data"][nd] = node_weight_per_segment
        
    return result
    
    
def getsimplepred(featuredict):
    result = {}
    for feature in featuredict:
        fdata = featuredict[feature]["data"]
        #nodecount = fdata.shape[0]
        #prediction np.zeros(nodecount, dtype=np.int64)
        prediction = np.argmax(fdata, axis=1)
        result.update({feature: prediction})
        
    return result




def get_argmax_and_confidence(arr):
    part = np.partition(arr, -2)
    first = part[-1]
    second = part[ -2]
    #print(np.argmax(arr), first/second)
    if second>0:
        conf = first/second
    else:
        conf = 100000
    return np.argmax(arr), conf


def save_features_and_prediction(fname, truth, prediction, longest, total, links):
    def prr(array):        
        result = ""
        for idx in range(array.shape[0]):
            result = result+" "+str(array[idx])        
        return result
        
    def brr(links, longest, total):
        result = f"{links[0]:.0f}({longest[0]:.2f}:{total[0]:.2f})" 
        for idx in range(1, links.shape[0]):
            result = result + ";" + f"{links[idx]:.0f}({longest[idx]:.2f}:{total[idx]:.2f})"
        return result
        
    
    with open(fname,"w") as f:
        stats = "lbl0"
        for idx in range(1, links[0].shape[0]):
            stats = stats+";lbl"+str(idx)
        f.write(f"idx;node_id;truth;predicted;error;{stats}\n")
        flag = "!"
        for nd in range(nodecount):            
            if truth[nd]!=prediction[nd]:                  
                stats = brr(links[nd], longest[nd], total[nd])
                f.write(f"{nd};{idxtranslator[nd]};{truth[nd]};{prediction[nd]};{flag};{stats}\n")
        flag = ""
        for nd in range(nodecount):            
            if truth[nd]==prediction[nd]:                  
                stats = brr(links[nd], longest[nd], total[nd])
                f.write(f"{nd};{idxtranslator[nd]};{truth[nd]};{prediction[nd]};{flag};{stats}\n")


def get_most_confident_prediction(featuredict, featureweights, featurepriority):    
    for feature in featurepriority:
        nodecount = featuredict[feature]["data"].shape[0]
        break
    
    prediction_hybrid = np.full(nodecount, -10, dtype=np.int64)
    #print (prediction_hybrid)
    for nd in range(nodecount):
        bestlabel, bestconfidence = -100, -100
        for feature in featurepriority:
            weight = featureweights[feature]
            if weight>0:
                arr = featuredict[feature]["data"]
                label, confidence = get_argmax_and_confidence(arr[nd] )
                if weight * confidence > bestconfidence:                    
                    bestlabel, bestconfidence = label, confidence

        prediction_hybrid[nd] = bestlabel            
    return prediction_hybrid




def without_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}

def combinationgenerator(featureweightrange):
    for feature in featureweightrange: #actually not a loop but "take first"
        excluding = without_keys(featureweightrange, feature)        
        if excluding != {}:
            for value in featureweightrange[feature]:                
                cg = combinationgenerator(excluding)                
                for combination in cg:                    
                    combination.update({feature:value})
                    yield combination                    
        else:            
            for value in featureweightrange[feature]:                
                yield {feature:value}
        break

        
        
def search_best_weights(featuredict, featureweightrange, featurepriority, labels, trainnodes = None, show_intermediate = False):
    best_acc = 0
    best_weights = {feature:-1 for feature in featuredict}
    for feature in featurepriority:
        nodecount = featuredict[feature]["data"].shape[0]
        break
    if not (trainnodes) is None:
        nodecount = trainnodes.shape[0]
    cg = combinationgenerator(featureweightrange)                   
    for featureweights in cg:
        if show_intermediate:
            print(featureweights)
        mostconfident = get_most_confident_prediction(featuredict, featureweights, featurepriority)
        if trainnodes is None:
            predictedlabels = mostconfident
        else:
            predictedlabels = mostconfident[trainnodes]
        acc = np.sum(labels == predictedlabels)/nodecount
        #todo maximize f1 macro instead
        if show_intermediate:
            print("current:", acc, "best", best_acc)
        if acc>best_acc:
            best_acc = acc
            best_weights = featureweights
    
    return best_weights, best_acc

def getrandompermutation(nodeclasses, rng):
    result = {}
    for c in nodeclasses:
        size = nodeclasses[c].shape[0]
        prmt = rng.permutation(size)
        result.update({c: prmt})
    return result

def dividetrainvaltest(nodeclasses, valshare, testshare, permutation = None):
    trainresult = {}
    valresult = {}
    testresult = {}
    for c in nodeclasses:
        if permutation is None:
            nodes = nodeclasses[c]
        else:
            perm = permutation[c]
            nodes = nodeclasses[c][perm]
        sep1 = round((1.-testshare-valshare)*nodes.shape[0])
        sep2 = round((1.-testshare)*nodes.shape[0])
        
        train = nodes[:sep1]
        val = nodes[sep1:sep2]
        test = nodes[sep2:]
        trainresult.update({c:train})
        valresult.update({c:val})
        testresult.update({c:test})
    return trainresult, valresult, testresult

def gettrainvaltestnodes(trainnodeclasses, valnodeclasses, testnodeclasses):
    train, val, test = [] , [] , []
    for c in trainnodeclasses:        
        train = train + list(trainnodeclasses[c])
        if not (valnodeclasses is None):
            val = val + list(valnodeclasses[c])
        if not (testnodeclasses is None):
            test = test + list(testnodeclasses[c])
    return np.array(train), np.array(val), np.array(test)



def checkpartition(G, trainnodeclasses, valnodeclasses, testnodeclasses, details=False, trns=None):
    '''
    check if some node (train, val or test) in partition lost all connections to train+val dataset
    
    also can be used to check original dataset: train=whole dataset, val=test=empty
    '''    
    is_ok = True
    dangling_train = 0
    dangling_val = 0
    dangling_test = 0
    errors = []
    trainnodes, valnodes, testnodes = gettrainvaltestnodes(trainnodeclasses, valnodeclasses, testnodeclasses)        
    trainvalnodes = np.concatenate((trainnodes, valnodes))
    
    for c in trainnodeclasses:
        trainnodes = trainnodeclasses[c]
        for node in trainnodes:
            node_ok = False
            for edge in nx.edge_boundary(G, [node], trainvalnodes):
                node_ok = True
                break
            if not node_ok:
                is_ok = False            
                msg = "Dangling train node from population " + c + ": " + str(node) 
                if not (trns is None):
                    msg = msg + ", original id:"+ str(trns[node])
                errors.append(msg)
                dangling_train += 1
                if details:
                    print(msg)
    if not (valnodeclasses is None):
        for c in valnodeclasses:
            valnodes = valnodeclasses[c]
            for node in valnodes:
                node_ok = False
                for edge in nx.edge_boundary(G, [node], trainvalnodes):
                    node_ok = True
                    break
                if not node_ok:
                    is_ok = False            
                    msg = "Dangling val node from population " + c + ": " + str(node) 
                    if not (trns is None):
                        msg = msg + ", original id:"+ str(trns[node])
                    errors.append(msg)
                    dangling_val += 1
                    if details:
                        print(msg)
    if not (testnodeclasses is None):
        for c in testnodeclasses:
            testnodes = testnodeclasses[c]
            for node in testnodes:
                node_ok = False
                for edge in nx.edge_boundary(G, [node], trainvalnodes):
                    node_ok = True
                    break
                if not node_ok:
                    is_ok = False            
                    msg = "Dangling test node from population " + c + ": " + str(node) 
                    if not (trns is None):
                        msg = msg + ", original id:"+ str(trns[node])
                    errors.append(msg)
                    dangling_test += 1
                    if details:
                        print(msg)

        
        
    if details:
        if is_ok:
            print("Partition is ok, no dangling nodes found")
        else:   
            dangling_total = dangling_train + dangling_val + dangling_test
            print(f"Bad partition, {dangling_train}:{dangling_val}:{dangling_test} dangling nodes found, total {dangling_total}")
    return is_ok, errors
