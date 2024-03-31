import torch
import torch.nn as nn
import torch.nn.functional as F

class POS_Encoder(nn.Module):
    def __init__(self, classes, hid1_dim,out_dim):
        super(POS_Encoder, self).__init__()
        self.classes = classes
        self.hid1_dim = hid1_dim
        self.out_dim = out_dim

        self.linear1 = torch.nn.Sequential(
            nn.Linear(self.classes, self.hid1_dim),
            nn.BatchNorm1d(self.hid1_dim),
            nn.PReLU(),
             )

        self.linear2 = torch.nn.Sequential(
                    nn.Linear(self.hid1_dim, self.out_dim),
                    nn.BatchNorm1d(self.out_dim),
                    nn.PReLU(),
             )

    def forward(self, pos_onehot):

        b_size, l_len, embed_dim = pos_onehot.shape
        pos_onehot = pos_onehot.view(-1, embed_dim)
        x = self.linear2(self.linear1(pos_onehot))
        x = x.view(b_size, l_len, -1)

        return x
