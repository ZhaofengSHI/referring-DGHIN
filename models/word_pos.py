import torch
import torch.nn as nn
import torch.nn.functional as F

class word_pos(nn.Module):
    def __init__(self, text_dim):
        super(word_pos, self).__init__()
        self.text_dim = text_dim
        # pw_fuse
        self.pw_fuse = torch.nn.Sequential(
            nn.Linear(self.text_dim, self.text_dim),
            nn.BatchNorm1d(self.text_dim),
            nn.PReLU()
        )

    def forward(self,words_map,pos_feature):
        # word pos feature fuse (element-wise product)
        pos_aware_words = words_map * pos_feature
        b_size, l_len, embed_dim = pos_aware_words.shape
        out = F.normalize(self.pw_fuse(pos_aware_words.view(-1,embed_dim)).view(b_size,l_len,-1),p=2,dim=2) + words_map

        return out