import torch
import torch.nn.functional as F
from torch.nn import init
import math
import torch as th
from torch.nn.parameter import Parameter
import torch.nn as nn

class get_softadj(nn.Module):
    def __init__(self, features):
        super(get_softadj, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(features, features))
        self.W2 = nn.Parameter(torch.FloatTensor(features, features))

        init.xavier_uniform_(self.W1)
        init.xavier_uniform_(self.W2)

    def forward(self, x):
        b,c,h,w = x.shape
        x = x.view(b,c,-1)
        g = torch.matmul(torch.matmul(self.W1, x).permute(0, 2, 1), torch.matmul(self.W2, x))
        g = F.softmax(g, dim=2)

        return x.permute(0,2,1), g

# create the visual GCN model
class GraphConvolution(nn.Module):
    """
    Simple pygGCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(th.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(th.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, infeatn, adj):

        support = th.einsum('bnd,df->bnf', (infeatn, self.weight))
        output = th.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN_V(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCN_V, self).__init__()
        self.conv1 = GraphConvolution(in_features, out_features)
        self.activate1 = nn.PReLU()

    def forward(self, x, A):
        x = self.conv1(x, A)
        x = self.activate1(x)

        return x

class visual_graph(nn.Module):
    def __init__(self, visual_dim):
        super(visual_graph, self).__init__()
        self.visual_dim = visual_dim
        self.node = torch.nn.Sequential(
            nn.Conv2d(self.visual_dim, self.visual_dim, kernel_size=1),
            nn.InstanceNorm2d(self.visual_dim, affine=True),
            nn.PReLU())
        self.get_adj = get_softadj(self.visual_dim)
        self.vis_graph = GCN_V(self.visual_dim,self.visual_dim)

    def forward(self, visual_feature):
        batch,channel,height,width = visual_feature.shape
        visual_node = self.node(visual_feature)
        node, adj = self.get_adj(visual_node)
        # graph reasoning
        graph_output = self.vis_graph(node, adj).permute(0, 2, 1).view(batch, -1, height, width)
        # residual
        out = F.normalize(graph_output,p=2,dim=1) + visual_feature

        return out
