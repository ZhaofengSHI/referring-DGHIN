import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
import torch

class BERTEncoder(nn.Module):
    def __init__(self, word_vec_size=768, weight=None):
        super(BERTEncoder, self).__init__()
        self.weight = weight
        self.text_encoder = BertModel.from_pretrained(self.weight)
        self.context_extractor = Context_Extractor(word_vec_size)

    def forward(self, word_id, attn_mask):

        words_feature = self.text_encoder(input_ids=word_id,attention_mask=attn_mask)[0]
        # extract context
        context = self.context_extractor(words_feature)

        return words_feature,context


class Context_Extractor(nn.Module):
    def __init__(self, input_dim):
        super(Context_Extractor, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.linear2 = torch.nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.PReLU())


    def forward(self,words):

        weight = self.linear(words).permute(0,2,1)
        # weighted sum
        weight = F.softmax(weight, dim=2)
        context = torch.bmm(weight,words).squeeze(1)
        context = self.linear2(context)

        return context


