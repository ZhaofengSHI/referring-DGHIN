import torch
import torch.nn as nn
import torch.nn.functional as F


class WordVisualAttention(nn.Module):

    def __init__(self, input_dim):
        super(WordVisualAttention, self).__init__()

        # compute features for att map
        self.visual_att = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.text_att = nn.Conv2d(input_dim,input_dim,kernel_size=1)

        #initialize pivot
        self.visual_pivot = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.text_pivot = nn.Conv2d(input_dim,input_dim,kernel_size=1)

        # weighted_lang feature
        self.weighted_lang = nn.Conv2d(input_dim,input_dim,kernel_size=1)

        # instance normalization
        self.in_norm1 = nn.InstanceNorm2d(input_dim, affine=True)
        self.in_norm2 = nn.InstanceNorm2d(input_dim, affine=True)

        # out_put layer
        self.out = torch.nn.Sequential(
            nn.Conv2d(input_dim,input_dim,kernel_size=1),
            nn.BatchNorm2d(input_dim),
            nn.PReLU())

    def forward(self, words, visual):

        words = words.permute(0, 2, 1).unsqueeze(-1)

        # compute feature for attn map
        words_att = self.text_att(words).squeeze(-1).permute(0,2,1)
        visual_attn = self.in_norm1(self.visual_att(visual))
        b_size, n_channel, h, w = visual_attn.shape
        visual_attn = visual_attn.view(b_size, n_channel, h*w)
        attn = torch.bmm(words_att, visual_attn)
        attn = F.softmax(attn,dim=1)

        # compute weighted lang
        words_pivot = self.text_pivot(words).squeeze(-1)
        weighted_emb = torch.bmm(words_pivot, attn)
        weighted_emb = weighted_emb.view(weighted_emb.size(0), weighted_emb.size(1), h, w)

        # compute visual pivot and weighted_lang
        weighted_lang = F.relu(self.in_norm2(self.weighted_lang(weighted_emb)),inplace=True)
        visual_pivot = F.relu(self.visual_pivot(visual),inplace=True)

        # fuse and output
        out_put = visual_pivot * weighted_lang
        out_put = self.out(out_put)

        return out_put
