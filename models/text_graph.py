import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
import torch.nn as nn
import torch.nn.functional as F

# create the GCN model(PyG)
class GCN_text(nn.Module):
    def __init__(self, in_c, out_c):
        super(GCN_text, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels=in_c, out_channels=out_c)
        self.activate1 = nn.PReLU()

    def forward(self, x, A):
        # data.x data.edge_index
        x = x  # [N, C]
        edge_index = A  # [2 ,E]
        hid = self.conv1(x=x, edge_index=edge_index)  # [N, D]
        out = self.activate1(hid)

        return out

def pyg_batch_builder(nodes_feature,edge_matrix):

    pyg_batch_data = []

    for i in range(nodes_feature.size(0)):

        edge_temp = edge_matrix[i,:,:]
        adj_coo = torch.nonzero(edge_temp).T.long()
        nodes_temp = nodes_feature[i,:,:]
        pyg_batch_data.append(Data(x=nodes_temp, edge_index=adj_coo))

    loader = DataLoader(pyg_batch_data, batch_size=nodes_feature.size(0), shuffle=False)
    pyg_batch = next(iter(loader))

    return pyg_batch


# create the GCN model(PyG)
class text_graph(nn.Module):
    def __init__(self, text_dim):
        super(text_graph, self).__init__()
        self.text_dim = text_dim
        # node project
        self.node = torch.nn.Sequential(
            nn.Linear(self.text_dim, self.text_dim),
            nn.BatchNorm1d(self.text_dim),
            nn.PReLU(),
             )
        self.text_gcn = GCN_text(self.text_dim,self.text_dim)

    def forward(self, text_feature, adj):
        b_size, l_len, embed_dim = text_feature.shape
        text_feature = text_feature.view(-1, embed_dim)
        text_node = self.node(text_feature).view(b_size, l_len, -1)
        text_feature = text_feature.view(b_size, l_len, -1)

        # build batch
        pyg_batch = pyg_batch_builder(text_node, adj)
        # graph reasoning
        graph_output = self.text_gcn(pyg_batch.x, pyg_batch.edge_index).view(b_size, l_len, -1)
        # residual
        out = F.normalize(graph_output,p=2,dim=2) + text_feature

        return out