import torch
from torch.nn import Linear, BatchNorm1d, LayerNorm, Sequential, LeakyReLU, Dropout
from torch_geometric.nn import GCNConv, TAGConv, GATv2Conv, TransformerConv, GMMConv, GINConv
from torch_geometric.nn.conv import SAGEConv
import torch.nn.functional as F
from mydata import num_classes


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BigAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()

        n_sub_graphs = 25
        feature_dim = 16

        self.conv_layers_1a = torch.nn.ModuleList([])
        self.batch_norms_1a = torch.nn.ModuleList([])
        self.conv_layers_1b = torch.nn.ModuleList([])
        self.batch_norms_1b = torch.nn.ModuleList([])

        self.conv_layers_2a = torch.nn.ModuleList([])
        self.batch_norms_2a = torch.nn.ModuleList([])
        self.conv_layers_2b = torch.nn.ModuleList([])
        self.batch_norms_2b = torch.nn.ModuleList([])

        self.big_norm = BatchNorm1d(feature_dim)

        for i in range(n_sub_graphs):
            self.conv_layers_1a.append(
                GATv2Conv(in_channels=5,
                          out_channels=feature_dim,
                          heads=2,
                          edge_dim=1,
                          aggr="mean",
                          concat=False,
                          share_weights=False,
                          add_self_loops=False)
            )

            self.batch_norms_1a.append(
                BatchNorm1d(feature_dim)
            )
            self.conv_layers_1b.append(
                GATv2Conv(in_channels=5,
                          out_channels=feature_dim,
                          heads=2,
                          edge_dim=1,
                          aggr="add",
                          concat=False,
                          share_weights=False,
                          add_self_loops=False)
            )

            self.batch_norms_1b.append(
                BatchNorm1d(feature_dim)
            )

            self.conv_layers_2a.append(
                GATv2Conv(in_channels=feature_dim,
                          out_channels=feature_dim,
                          heads=2,
                          edge_dim=1,
                          aggr="mean",
                          concat=False,
                          share_weights=False,
                          add_self_loops=True)
            )

            self.batch_norms_2a.append(
                BatchNorm1d(feature_dim)
            )

            self.conv_layers_2b.append(
                GATv2Conv(in_channels=feature_dim,
                          out_channels=feature_dim,
                          heads=2,
                          edge_dim=1,
                          aggr="add",
                          concat=False,
                          share_weights=False,
                          add_self_loops=True)
            )

            self.batch_norms_2b.append(
                BatchNorm1d(feature_dim)
            )

        fc_dim = 2 * n_sub_graphs * feature_dim
        self.fc1 = Linear(fc_dim, fc_dim)
        self.norm1 = BatchNorm1d(fc_dim)
        self.fc2 = Linear(fc_dim, fc_dim)
        self.norm2 = BatchNorm1d(fc_dim)
        self.fc3 = Linear(fc_dim, 5)

    def forward(self, x_input, bf, sub_data_25, train_edge_index, train_edge_weight):
        res1 = []

        for i in range(25):
            edge_index, edge_weight = sub_data_25[i]

            h = self.conv_layers_1a[i](x_input, edge_index, edge_weight)
            h = self.batch_norms_1a[i](h)
            h = F.leaky_relu(h)

            h = self.conv_layers_2a[i](h, edge_index, edge_weight)
            h = self.batch_norms_2a[i](h)
            h = F.leaky_relu(h)

            res1.append(h)

            h = self.conv_layers_1b[i](x_input, edge_index, edge_weight)
            h = self.batch_norms_1b[i](h)
            h = F.leaky_relu(h)

            h = self.conv_layers_2b[i](h, edge_index, edge_weight)
            h = self.batch_norms_2b[i](h)
            h = F.leaky_relu(h)

            res1.append(h)

        h = torch.cat(res1, dim=-1)
        h = self.norm1(self.fc1(h)).relu()
        h = self.norm2(self.fc2(h)).relu()
        h = self.fc3(h)
        return h


class AttnGCN(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        n_features = 128
        n_heads = 2
        self.dp = 0.2

        n_class = num_classes[dataset]

        self.conv1 = GATv2Conv(in_channels=n_class,
                               out_channels=n_features,
                               heads=n_heads,
                               edge_dim=1,
                               aggr="add",
                               concat=True,
                               share_weights=False,
                               add_self_loops=False)
        self.norm1 = BatchNorm1d(n_features * n_heads)

        self.conv2 = GATv2Conv(in_channels=n_features * n_heads,
                               out_channels=n_features,
                               heads=n_heads,
                               edge_dim=1,
                               aggr="add",
                               concat=True,
                               share_weights=False,
                               add_self_loops=True)

        self.norm2 = BatchNorm1d(n_features * n_heads)
        self.fc = Linear(n_features * n_heads, n_class)

    #         self.conv3 = GATv2Conv(in_channels=n_features,
    #                                out_channels=5,
    #                                heads=1,
    #                                edge_dim=1,
    #                                aggr="add",
    #                                concat=False,
    #                                share_weights=False,
    #                                add_self_loops=True)
    #         self.norm3 = BatchNorm1d(n_features)

    def forward(self, x_input, edge_index, edge_weight):
        h = self.conv1(x_input, edge_index, edge_weight)
        h = self.norm1(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dp, training=True)

        h = self.conv2(h, edge_index, edge_weight)
        h = self.norm2(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dp, training=True)

        h = self.fc(h)
        return h


class SimpleNN(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        dp = 0.2
        hidden_dim = 128
        n_class = num_classes[dataset]
        self.model = Sequential(
            Linear(3 * n_class, hidden_dim),
            BatchNorm1d(hidden_dim),
            LeakyReLU(),
            Dropout(p=dp),
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            LeakyReLU(),
            Dropout(p=dp),
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            LeakyReLU(),
            Dropout(p=dp),
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            LeakyReLU(),
            Dropout(p=dp),
            Linear(hidden_dim, n_class),
        )

    def forward(self, h, a, b):
        h = self.model(h)
        return h


class GCN(torch.nn.Module):
    def __init__(self, dataset):
        super(GCN, self).__init__()
        hidden_dim = 128
        n_class = num_classes[dataset]
        # First GCN layer with normalization
        self.conv1 = GCNConv(n_class, hidden_dim, normalize=True)

        # Second GCN layer with normalization
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=True)

        # Output layer
        self.fc = torch.nn.Linear(hidden_dim, n_class)

    def forward(self, x, edge_index, edge_weight):
        # Apply the first GCN layer
        x = F.relu(self.conv1(x, edge_index, edge_weight))

        # Apply the second GCN layer
        x = F.relu(self.conv2(x, edge_index, edge_weight))

        # Fully connected layer for classification
        x = self.fc(x)

        return x


class GCN_simple(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.norm1 = BatchNorm1d(5)

        self.attn_conv = GATv2Conv(in_channels=5,
                                   out_channels=128,
                                   heads=2,
                                   edge_dim=1,
                                   aggr="mean",
                                   concat=False,
                                   share_weights=False,
                                   add_self_loops=False
                                   )
        self.attn_norm = BatchNorm1d(128)

        self.fc1 = Linear(128, 128)
        self.norm_fc1 = BatchNorm1d(128)
        self.fc2 = Linear(128, 5)
        self.dp = 0.2

    def forward(self, h, edge_index, edge_weight):
        h = self.norm1(h)

        h = self.attn_conv(h, edge_index, edge_weight)

        h = self.fc1(h)
        h = self.norm_fc1(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dp, training=self.training)

        h = self.fc2(h)

        return h


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n_features = 128
        n_heads = 1
        self.dp = 0.2
        self.conv1 = TransformerConv(in_channels=15,
                                     out_channels=n_features,
                                     heads=n_heads,
                                     concat=True,
                                     beta=False,
                                     dropout=0.2,
                                     edge_dim=1,
                                     bias=True,
                                     root_weight=False,
                                     )
        self.norm1 = BatchNorm1d(n_features * n_heads)

        self.conv2 = TransformerConv(in_channels=n_features * n_heads,
                                     out_channels=n_features * n_heads,
                                     heads=n_heads,
                                     concat=True,
                                     beta=False,
                                     dropout=0.2,
                                     edge_dim=1,
                                     bias=True,
                                     root_weight=True,
                                     )
        self.norm2 = BatchNorm1d(n_features * n_heads)

        self.conv3 = TransformerConv(in_channels=n_features * n_heads,
                                     out_channels=5,
                                     heads=n_heads,
                                     concat=False,
                                     beta=False,
                                     dropout=0.2,
                                     edge_dim=1,
                                     bias=True,
                                     root_weight=True,
                                     )
        self.norm3 = BatchNorm1d(5)

    def forward(self, x_input, bf, sub_data_25, edge_index, edge_weight):
        h, t = self.conv1(bf, edge_index, edge_weight, return_attention_weights=True)
        _, edge_weight = t
        h = self.norm1(h)
        h = F.leaky_relu(h)

        h, t = self.conv2(h, edge_index, edge_weight, return_attention_weights=True)
        _, edge_weight = t
        h = self.norm2(h)
        h = F.leaky_relu(h)

        h = self.conv3(h, edge_index, edge_weight)
        return h


class TAGConv_3l_512h_w_k3(torch.nn.Module):
    def __init__(self, dataset):
        super(TAGConv_3l_512h_w_k3, self).__init__()
        n_class = num_classes[dataset]
        self.conv1 = TAGConv(n_class, 128)
        self.conv2 = TAGConv(128, 128)
        self.conv3 = TAGConv(128, n_class)

    def forward(self, x, edge_index, edge_weight):
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.elu(self.conv2(x, edge_index, edge_weight))
        x = self.conv3(x, edge_index, edge_weight)
        return x


from torch_geometric.nn import GINConv, global_add_pool


class GINNet(torch.nn.Module):
    def __init__(self, dataset):
        super(GINNet, self).__init__()
        n_class = num_classes[dataset]
        hidden_dim = 128
        # GIN Convolution Layer
        self.conv1 = GINConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(n_class, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU()
            ),
            eps=0.0  # Add a small value to the denominator for numerical stability
        )

        self.conv2 = GINConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU()
            ),
            eps=0.0
        )

        # Output layer
        self.fc = torch.nn.Linear(hidden_dim, n_class)

    def forward(self, x, edge_index, edge_weight):
        # Apply GIN Convolution layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Fully connected layer for classification
        x = self.fc(x)

        return F.log_softmax(x, dim=1)
