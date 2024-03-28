import os
import time
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import networkx as nx
import torch.nn as nn
from sklearn import metrics
from numba import njit, prange
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import bernoulli
from sklearn.model_selection import train_test_split
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import InMemoryDataset, Data
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, NNConv, SGConv, ARMAConv, TAGConv, ChebConv, DNAConv, \
EdgeConv, FiLMConv, FastRGCNConv, SSGConv, SAGEConv, GATv2Conv, BatchNorm, GraphNorm, MemPooling, SAGPooling


def symmetrize(m):
    return m + m.T - np.diag(m.diagonal())


def generate_matrices_fn(population_sizes, offset, edge_probs, mean_weight, rng):
    '''
        main simulation function
    
    Parameters
    ----------
    population_sizes: list 
        list of population sizes
    offset: float
        we assume ibdsum pdf = lam*exp(-lam*(x-offset)) for x>offset and 0 otherwise, lam = 1/mean
    edge_probs: 2d array
        probability of an edge between classes
    mean_weight: 2d array
        mean weight of an existing edge between classes (corrected by offset)
    rng: random number generator
    
    Returns
    -------
    counts: 
        
    sums: 
        
    pop_index: 1d np array
        population index of every node
        
    '''
    p = edge_probs
    teta = mean_weight
    pop_index = []
    n_pops = len(population_sizes)
    for i in range(n_pops):
        pop_index += [i] * population_sizes[i]

    pop_index = np.array(pop_index)
    print(f"{n_pops=}")
    blocks_sums = [[np.zeros(shape=(population_sizes[i], population_sizes[j])) for j in range(n_pops)] for i in
                   range(n_pops)]
    blocks_counts = [[np.zeros(shape=(population_sizes[i], population_sizes[j])) for j in range(n_pops)] for i
                     in range(n_pops)]

    #print(np.array(blocks_sums).shape)

    for pop_i in range(n_pops):
        for pop_j in range(pop_i + 1):
            if p[pop_i, pop_j] == 0:
                continue
            print(f"{pop_i=} {pop_j=}")
            pop_cross = population_sizes[pop_i] * population_sizes[pop_j]
            #TODO switch to rng.binomial or something
            bern_samples = bernoulli.rvs(p[pop_i, pop_j], size=pop_cross)
            total_segments = np.sum(bern_samples)
            print(f"{total_segments=}")
            exponential_samples = rng.exponential(teta[pop_i, pop_j], size=total_segments) + offset
            #position = 0
            exponential_totals_samples = np.zeros(pop_cross, dtype=np.float64)
            #mean_totals_samples = np.zeros(pop_cross, dtype=np.float64)
            exponential_totals_samples[bern_samples == 1] = exponential_samples

            bern_samples = np.reshape(bern_samples, newshape=(population_sizes[pop_i], population_sizes[pop_j]))
            exponential_totals_samples = np.reshape(exponential_totals_samples,
                                                    newshape=(population_sizes[pop_i], population_sizes[pop_j]))
            if (pop_i == pop_j):
                bern_samples = np.tril(bern_samples, -1)
                exponential_totals_samples = np.tril(exponential_totals_samples, -1)
            blocks_counts[pop_i][pop_j] = bern_samples
            blocks_sums[pop_i][pop_j] = exponential_totals_samples
    
    
    full_blocks_counts = np.block(blocks_counts)
    full_blocks_sums = np.block(blocks_sums)
    return np.nan_to_num(symmetrize(full_blocks_counts)), np.nan_to_num(symmetrize(full_blocks_sums)), pop_index


def simulate_graph_fn(classes, means, counts, pop_index, path):
    '''
        store simulated dataframe
    
    Parameters
    ----------
    classes: list of str
        names of populations
    means: 2d np array
        0: no link between i-th and j-th individuals
    counts: 2d np array
        ibd sum between i-th and j-th individuals
    pop_index: 1d np array
        population index of every node
    path: string
        csv file to store dataframe
    '''
    indiv = list(range(counts.shape[0]))
    with open(path, 'w', encoding="utf-8") as f:
        f.write('node_id1,node_id2,label_id1,label_id2,ibd_sum,ibd_n\n')
        for i in range(counts.shape[0]):
            for j in range(i):
                if (counts[i][j]):
                    name_i = classes[pop_index[i]] if "," not in classes[pop_index[i]] else '\"' + classes[pop_index[i]] + '\"'
                    name_j = classes[pop_index[j]] if "," not in classes[pop_index[j]] else '\"' + classes[pop_index[j]] + '\"'
                    f.write(f'node_{i},node_{j},{name_i},{name_j},{means[i][j]},{counts[i][j]}\n')



class DataProcessor:
    def __init__(self, path):
        self.dataset_name: str = None
        self.train_size: float = None
        self.valid_size: float = None
        self.test_size: float = None
        self.edge_probs = None
        self.mean_weight = None
        self.offset = 8.0
        self.df = pd.read_csv(path)
        self.node_names_to_int_mapping: dict[str, int] = self.get_node_names_to_int_mapping(self.get_unique_nodes(self.df))
        self.classes: list[str] = self.get_classes(self.df)
        self.node_classes_sorted: pd.DataFrame = self.get_node_classes(self.df)
        self.class_to_int_mapping: dict[int, str] = {i:n for i, n in enumerate(self.classes)}
        self.train_nodes = None
        self.valid_nodes = None
        self.test_nodes = None
        self.array_of_graphs_for_training = []
        self.array_of_graphs_for_validation = []
        self.array_of_graphs_for_testing = []
        self.rng = np.random.default_rng(seed)
        
    def get_classes(self, df):
        # return ['карачаевцы,балкарцы', 'осетины', 'кабардинцы,черкесы,адыгейцы','ингуши','кумыки','ногайцы','чеченцы','дагестанские народы']
        return pd.concat([df['label_id1'], df['label_id2']], axis=0).unique().tolist()

    def get_unique_nodes(self, df):
        return pd.concat([df['node_id1'], df['node_id2']], axis=0).drop_duplicates().to_numpy()

    def get_node_names_to_int_mapping(self, unique_nodes):
        # d = torch.load(r"C:\HSE\genotek-nationality-analysis\data\mapping_indices.pt")
        # d = {'node_'+str(k):v for k, v in d.items()}
        # return d
        return {n:i for i, n in enumerate(unique_nodes)}

    def relabel(self, df):
        '''
        Replace any node names with continious int numbers
        :param df: initial DataFrame
        :return: same DataFrame but with new labels
        '''
        df.iloc[:, 0] = df.iloc[:, 0].apply(lambda n: self.node_names_to_int_mapping[n])
        df.iloc[:, 1] = df.iloc[:, 1].apply(lambda n: self.node_names_to_int_mapping[n])
        return df

    def class_labels_to_int(self, df):
        df.iloc[:, 2] = df.iloc[:, 2].apply(lambda t: self.classes.index(t))
        df.iloc[:, 3] = df.iloc[:, 3].apply(lambda t: self.classes.index(t))
        return df

    def get_node_classes(self, df):
        self.df = self.relabel(self.df)
        self.df = self.class_labels_to_int(self.df)

        n = pd.concat([df['node_id1'], df['node_id2']], axis=0)

        l = pd.concat([df['label_id1'], df['label_id2']], axis=0)

        df_node_classes = pd.concat([n, l], axis=1).drop_duplicates()

        df_node_classes.columns = ['node', 'class_id']

        return df_node_classes.sort_values(by=['node'])

    def node_classes_to_dict(self):
        return {n: c for index, pair in self.node_classes_sorted.iterrows() for n, c in [pair.tolist()]}
        
    def generate_random_train_valid_test_nodes(self, train_size, valid_size, test_size, random_state, save_dir=None):
        if train_size + valid_size + test_size != 1.0:
            raise Exception("All sizes should add up to 1.0!")

        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        num_nodes_per_class = self.node_classes_sorted.iloc[:, 1].value_counts()
        node_classes_random = self.node_classes_sorted.sample(frac=1, random_state=random_state)
        self.train_nodes, self.valid_nodes, self.test_nodes = [], [], []
        node_counter = {i: 0 for i in range(num_nodes_per_class.shape[0])}

        for i in range(node_classes_random.shape[0]):
            node_class = node_classes_random.iloc[i, 1]
            if node_counter[node_class] <= int(self.train_size * num_nodes_per_class.loc[node_class]):
                self.train_nodes.append(node_classes_random.iloc[i, 0])
                node_counter[node_class] += 1
            elif int((self.train_size + self.valid_size) * num_nodes_per_class.loc[node_class]) >= node_counter[node_class] > int(self.train_size * num_nodes_per_class.loc[node_class]):
                self.valid_nodes.append(node_classes_random.iloc[i, 0])
                node_counter[node_class] += 1
            else:
                self.test_nodes.append(node_classes_random.iloc[i, 0])

        if save_dir is not None:
            with open(save_dir + '/train.pickle', 'wb') as handle:
                pickle.dump(self.train_nodes, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(save_dir + '/valid.pickle', 'wb') as handle:
                pickle.dump(self.valid_nodes, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(save_dir + '/test.pickle', 'wb') as handle:
                pickle.dump(self.test_nodes, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def load_train_valid_test_nodes(self, train_path, valid_path, test_path, data_type):
        if data_type == 'pickle':
            with open(train_path, 'rb') as handle:
                self.train_nodes = pickle.load(handle)
            with open(valid_path, 'rb') as handle:
                self.valid_nodes = pickle.load(handle)
            with open(test_path, 'rb') as handle:
                self.test_nodes = pickle.load(handle)
        if data_type == 'torch':
            self.train_nodes = torch.load(train_path).tolist()
            self.valid_nodes = torch.load(valid_path).tolist()
            self.test_nodes = torch.load(test_path).tolist()

        if not (type(self.train_nodes) == list and type(self.valid_nodes) == list and type(self.test_nodes) == list):
            raise Exception('Node ids must be stored in Python lists!')
        if len(set(self.train_nodes + self.valid_nodes + self.test_nodes)) < (len(self.train_nodes) + len(self.valid_nodes) + len(self.test_nodes)):
            raise Exception('There is intersection between train, valid and test node sets!')

    def place_specific_node_to_the_end(self, node_list, node_id):
        curr_node = node_list[node_id]
        new_node_list = node_list + [curr_node]
        new_node_list.remove(curr_node)  # remove node from the beginning and leave at the end

        return new_node_list, curr_node

    def make_hashmap(self, nodes):
        hashmap = np.array([1e3 for i in range(self.node_classes_sorted.shape[0])]).astype(int)
        for i, node in enumerate(nodes):
            hashmap[node] = i

        return hashmap

    def make_one_hot_encoded_features(self, all_nodes, specific_nodes, hashmap, dict_node_classes):
        # order of features is the same as order nodes in self.nodes
        features = np.zeros((len(all_nodes), len(self.classes)))
        for n in all_nodes:
            if n in specific_nodes:
                features[hashmap[int(n)], :] = [1 / len(self.classes)] * len(self.classes)
            else:
                features[hashmap[int(n)], :] = [1 if i == dict_node_classes[n] else 0 for i in range(len(self.classes))]

        return features

    def construct_node_classes(self, nodes, dict_node_classes):
        targets = []
        for node in nodes:
            targets.append(dict_node_classes[node])

        return targets

    @staticmethod
    @njit(cache=True)
    def drop_rows_for_training_dataset(df, test_nodes):
        drop_rows = []
        for i in range(df.shape[0]):  # speed it up in future
            row = df[i, :]
            if int(row[0]) in test_nodes or int(row[1]) in test_nodes:
                drop_rows.append(i)

        return drop_rows

    @staticmethod
    @njit(cache=True)
    def construct_edges(df, hashmap):

        weighted_edges = []

        for i in range(df.shape[0]):
            row = df[i]
            weighted_edges.append([hashmap[int(row[0])], hashmap[int(row[1])], row[4]])
            weighted_edges.append([hashmap[int(row[1])], hashmap[int(row[0])], row[4]])

        return np.array(weighted_edges)

    @staticmethod
    @njit(cache=True)
    def find_connections_to_nodes(df, train_nodes, non_train_nodes):

        rows_for_adding_per_node = []

        for i in range(len(non_train_nodes)):
            tmp = []
            for j in range(df.shape[0]):
                row = df[j]
                if int(row[0]) == non_train_nodes[i] and int(row[1]) in train_nodes or int(row[1]) == non_train_nodes[i] and int(
                        row[0]) in train_nodes:
                    tmp.append(j)

            rows_for_adding_per_node.append(tmp)

        return rows_for_adding_per_node

    def generate_graph(self, curr_nodes, specific_node, dict_node_classes, df):

        hashmap = self.make_hashmap(curr_nodes)
        features = self.make_one_hot_encoded_features(curr_nodes, [specific_node], hashmap,
                                                      dict_node_classes)
        targets = self.construct_node_classes(curr_nodes, dict_node_classes)
        weighted_edges = self.construct_edges(df.to_numpy(), hashmap)

        # sort edges
        sort_idx = np.lexsort((weighted_edges[:, 1], weighted_edges[:, 0]))
        weighted_edges = weighted_edges[sort_idx]

        graph = Data.from_dict(
            {'y': torch.tensor(targets, dtype=torch.long), 'x': torch.tensor(features),
             'weight': torch.tensor(weighted_edges[:, 2]),
             'edge_index': torch.tensor(weighted_edges[:, :2].T, dtype=torch.long)})

        graph.num_classes = len(self.classes)

        return graph

    def make_train_valid_test_datasets_with_numba(self, feature_type, model_type, train_dataset_type, test_dataset_type, dataset_name):

        self.dataset_name = dataset_name

        self.array_of_graphs_for_training = []
        self.array_of_graphs_for_testing = []
        self.array_of_graphs_for_validation = []

        if feature_type == 'one_hot' and model_type == 'homogeneous':
            if train_dataset_type == 'multiple' and test_dataset_type == 'multiple':
                dict_node_classes = self.node_classes_to_dict()
                df_for_training = self.df.copy()
                drop_rows = self.drop_rows_for_training_dataset(self.df.to_numpy(), self.valid_nodes + self.test_nodes)
                df_for_training = df_for_training.drop(drop_rows)

                # make training samples
                for k in tqdm(range(len(self.train_nodes)), desc='Make train samples'):
                    curr_train_nodes, specific_node = self.place_specific_node_to_the_end(self.train_nodes, k)

                    graph = self.generate_graph(curr_train_nodes, specific_node, dict_node_classes, df_for_training)

                    self.array_of_graphs_for_training.append(graph)

                # make validation samples
                rows_for_adding_per_node = self.find_connections_to_nodes(self.df.to_numpy(),
                                                                               np.array(self.train_nodes),
                                                                               np.array(self.valid_nodes))
                for k in tqdm(range(len(self.valid_nodes)), desc='Make valid samples'):
                    rows_for_adding = rows_for_adding_per_node[k]
                    df_for_validation = pd.concat([df_for_training, self.df.iloc[rows_for_adding]], axis=0)

                    if df_for_validation.shape[0] == df_for_training.shape[0]:
                        print('Isolated val node found! Restart with different seed or this node will be ignored.')
                        continue

                    specific_node = self.valid_nodes[k]
                    current_valid_nodes = self.train_nodes + [specific_node]

                    graph = self.generate_graph(current_valid_nodes, specific_node, dict_node_classes, df_for_validation)

                    self.array_of_graphs_for_validation.append(graph)

                # make testing samples
                rows_for_adding_per_node = self.find_connections_to_nodes(self.df.to_numpy(),
                                                                               np.array(self.train_nodes),
                                                                               np.array(self.test_nodes))
                for k in tqdm(range(len(self.test_nodes)), desc='Make test samples'):
                    rows_for_adding = rows_for_adding_per_node[k]
                    df_for_testing = pd.concat([df_for_training, self.df.iloc[rows_for_adding]], axis=0)

                    if df_for_testing.shape[0] == df_for_training.shape[0]:
                        print('Isolated test node found! Restart with different seed or this node will be ignored.')
                        continue

                    specific_node = self.test_nodes[k]
                    current_test_nodes = self.train_nodes + [specific_node]

                    graph = self.generate_graph(current_test_nodes, specific_node, dict_node_classes, df_for_testing)

                    self.array_of_graphs_for_testing.append(graph)

    def compute_simulation_params(self):
        self.edge_probs = np.zeros((len(self.classes), len(self.classes)))
        self.mean_weight = np.zeros((len(self.classes), len(self.classes)))
        for i in range(len(self.classes)):
            for j in range(len(self.classes)):
                if i == j:
                    real_connections_df = self.df[(self.df.label_id1 == i) & (self.df.label_id2 == j)]
                else:
                    real_connections_df = self.df[
                        ((self.df.label_id1 == i) & (self.df.label_id2 == j)) | (
                                    (self.df.label_id1 == j) & (self.df.label_id2 == i))]
                real_connections = real_connections_df.shape[0]
                num_nodes = len(
                    pd.concat([real_connections_df['node_id1'], real_connections_df['node_id2']], axis=0).unique())

                self.mean_weight[i, j] = real_connections_df['ibd_sum'].to_numpy().mean() - self.offset
                if np.isnan(self.mean_weight[i, j]):
                    self.mean_weight[i, j] = -self.offset

                if i == j:
                    all_possible_connections = num_nodes * (num_nodes - 1) / 2
                else:
                    n = pd.concat([self.df['node_id1'], self.df['node_id2']], axis=0)
                    l = pd.concat([self.df['label_id1'], self.df['label_id2']], axis=0)
                    df_new = pd.concat([n, l], axis=1)
                    df_new = df_new.drop_duplicates()
                    num_nodes_class_1 = len(df_new.iloc[:, 1][df_new.iloc[:, 1] == i].to_numpy())
                    num_nodes_class_2 = len(df_new.iloc[:, 1][df_new.iloc[:, 1] == j].to_numpy())
                    all_possible_connections = num_nodes_class_1 * num_nodes_class_2
                self.edge_probs[i, j] = real_connections / all_possible_connections

    

    def generate_matrices(self, population_sizes):
        return generate_matrices_fn(population_sizes, self.offset, self.edge_probs, self.mean_weight, self.rng)
        
    def simulate_graph(self, means, counts, pop_index, path):
        simulate_graph_fn(self.classes, means, counts, pop_index, path)        
        # remove isolated nodes
        # G.remove_nodes_from(list(nx.isolates(G)))



class Trainer:
    def __init__(self, data: DataProcessor, model_cls, lr, wd, loss_fn, weight, batch_size, log_dir, patience, num_epochs):
        self.data = data
        self.model = None
        self.device = None
        self.model_cls = model_cls
        self.learning_rate = lr
        self.weight_decay = wd
        self.loss_fn = loss_fn
        self.weight = weight
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.patience = patience
        self.num_epochs = num_epochs
        self.max_f1_score_macro = 0
        self.patience_counter = 0

    def compute_metrics_cross_entropy(self, graphs):
        y_true = []
        y_pred = []

        for i in tqdm(range(len(graphs)), desc='Compute metrics'):
            p = F.softmax(self.model(graphs[i].to(self.device))[-1],
                          dim=0).cpu().detach().numpy()
            y_pred.append(np.argmax(p))
            y_true.append(graphs[i].y[-1].cpu().detach())
            graphs[i].to('cpu')

        return y_true, y_pred

    def evaluation(self, i):
        self.model.eval()

        y_true, y_pred = self.compute_metrics_cross_entropy(self.data.array_of_graphs_for_validation)

        print('Evaluation report')
        print(classification_report(y_true, y_pred))

        current_f1_score_macro = f1_score(y_true, y_pred, average='macro')
        if current_f1_score_macro > self.max_f1_score_macro:
            self.max_f1_score_macro = current_f1_score_macro
            print(f'f1 macro improvement to {self.max_f1_score_macro}')
            torch.save(self.model.state_dict(), self.log_dir + '/model_best.bin')
        else:
            self.patience_counter += 1
            print(f'Metric was not improved for the {self.patience_counter}th time')

    def test(self):
        self.model = self.model_cls(self.data.array_of_graphs_for_training[0]).to(self.device)
        self.model.load_state_dict(torch.load(self.log_dir + '/model_best.bin'))
        self.model.eval()
        y_true, y_pred = self.compute_metrics_cross_entropy(self.data.array_of_graphs_for_testing)
        print('Test report')
        print(classification_report(y_true, y_pred))
        print(f"f1 macro score on test dataset: {f1_score(y_true, y_pred, average='macro')}")

        cm = confusion_matrix(y_true, y_pred)

        plt.clf()
        fig, ax = plt.subplots(1, 1)
        sns.heatmap(cm, annot=True, fmt=".2f", ax=ax)
        plt.show()

    def run(self):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model_cls(self.data.array_of_graphs_for_training[0]).to(self.device) # just initialize the parameters of the model
        criterion = self.loss_fn(weight=self.weight)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=500, gamma=0.1)
        print(f'Training for data: {self.data.dataset_name}')
        self.max_f1_score_macro = 0
        self.patience_counter = 0

        if self.loss_fn == torch.nn.CrossEntropyLoss:
            for i in tqdm(range(self.num_epochs), desc='Training epochs'):
                if self.patience_counter == self.patience:
                    break
                self.evaluation(i)

                self.model.train()

                selector = np.array([i for i in range(len(self.data.array_of_graphs_for_training))])
                np.random.shuffle(selector)

                mean_epoch_loss = []

                pbar = tqdm(range(len(selector)), desc='Training samples')
                pbar.set_postfix({'val_best_score': self.max_f1_score_macro})
                for j in pbar:
                    n = selector[j]
                    data_curr = self.data.array_of_graphs_for_training[n].to(self.device)
                    optimizer.zero_grad()
                    out = self.model(data_curr)
                    loss = criterion(out[-1], data_curr.y[-1])
                    loss.backward()
                    mean_epoch_loss.append(loss.detach().cpu().numpy())
                    optimizer.step()
                    scheduler.step()
                    self.data.array_of_graphs_for_training[n].to('cpu')

                y_true, y_pred = self.compute_metrics_cross_entropy(self.data.array_of_graphs_for_training)

                print('Training report')
                print(classification_report(y_true, y_pred))

            self.test()


class TAGConv_3l_128h_w_k3(torch.nn.Module):
    def __init__(self, data):
        super(TAGConv_3l_128h_w_k3, self).__init__()
        self.conv1 = TAGConv(data.num_features, 128)
        self.conv2 = TAGConv(128, 128)
        self.conv3 = TAGConv(128, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        return x


class TAGConv_3l_512h_w_k3(torch.nn.Module):
    def __init__(self, data):
        super(TAGConv_3l_512h_w_k3, self).__init__()
        self.conv1 = TAGConv(data.num_features, 512)
        self.conv2 = TAGConv(512, 512)
        self.conv3 = TAGConv(512, int(data.num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        return x







