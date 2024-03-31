import torch
from models.visual_backbone import swin_base_patch4_window12_384_in22k as swin_backbone
import torch.nn as nn
from models.word_pos import word_pos
from models.utils import generate_coord
import torch.nn.functional as F
from models.text_backbone import BERTEncoder
from models.VisualWordAtt import WordVisualAttention
from models.pos_encoder import POS_Encoder
from models.visual_graph import visual_graph
from models.text_graph import text_graph

def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=False)

class ConvBNRelu(nn.Module):
    # decoder conv module
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,  use_relu=True):
        super(ConvBNRelu, self).__init__()
        self.use_relu = use_relu
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.use_relu:
            self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x

class model(nn.Module):

    def __init__(self,visual_dim = 512, text_dim = 512, graph_dim = 512, config=None):
        super(model, self).__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.graph_dim = graph_dim

        self.max_len = config.word_len
        self.bert_weight = config.BERT_weight


        self.img_bn = nn.BatchNorm2d(3)

        self.v_0_conv2 = ConvBNRelu(128 + 128 + 8, self.visual_dim, kernel_size=3, stride=1, padding=1)

        self.v_1_conv2 = ConvBNRelu(256 + 8, self.visual_dim, kernel_size=3, stride=1, padding=1)

        self.v_2_conv2 = ConvBNRelu(512 + 8, self.visual_dim, kernel_size=3, stride=1, padding=1)

        self.v_3_conv2 = ConvBNRelu(1024 + 8, self.visual_dim, kernel_size=3, stride=1, padding=1)

        # ht mapping
        self.mapping_hT = torch.nn.Sequential(
            nn.Linear(768, 128),
            nn.BatchNorm1d(128),
            nn.PReLU()
        )

        # words mapping
        self.mapping_word = torch.nn.Sequential(
            nn.Linear(768, self.text_dim),
            nn.BatchNorm1d(self.text_dim),
            nn.PReLU()
        )

        # pos encoder
        self.pos_encoder = POS_Encoder(7, self.text_dim * 2, self.text_dim)#

        # word pos feature fuse
        self.word_pos = word_pos(self.text_dim)#

        # word visual attention
        self.wordvisual1 = WordVisualAttention(512)
        self.wordvisual2 = WordVisualAttention(512)
        self.wordvisual3 = WordVisualAttention(512)

        # text graph
        self.text_graph_1 = text_graph(self.graph_dim)
        self.text_graph_2 = text_graph(self.graph_dim)
        self.text_graph_3 = text_graph(self.graph_dim)

        # visual graph
        self.vis_graph_1 = visual_graph(self.graph_dim)
        self.vis_graph_2 = visual_graph(self.graph_dim)
        self.vis_graph_3 = visual_graph(self.graph_dim)

        # graph word visual attention
        self.graph_wordvisual1 = WordVisualAttention(512)
        self.graph_wordvisual2 = WordVisualAttention(512)
        self.graph_wordvisual3 = WordVisualAttention(512)

        # pre-graph post-graph fuse
        self.fea_fuse_1 = ConvBNRelu(self.visual_dim * 2, self.visual_dim, kernel_size=3, stride=1, padding=1)
        self.fea_fuse_2 = ConvBNRelu(self.visual_dim * 2, self.visual_dim, kernel_size=3, stride=1, padding=1)
        self.fea_fuse_3 = ConvBNRelu(self.visual_dim * 2, self.visual_dim, kernel_size=3, stride=1, padding=1)

        #####fuse
        self.fuse_2 = ConvBNRelu(self.visual_dim * 2, self.visual_dim, kernel_size=3, stride=1, padding=1)
        self.fuse_1 = ConvBNRelu(self.visual_dim * 2, self.visual_dim, kernel_size=3, stride=1, padding=1)
        self.fuse_0 = ConvBNRelu(self.visual_dim * 2, self.visual_dim, kernel_size=3, stride=1, padding=1)

        self.seg_out = nn.Sequential(
            ConvBNRelu(self.visual_dim, self.visual_dim // 4, kernel_size=3, stride=1, padding=1),
            ConvBNRelu(self.visual_dim // 4, self.visual_dim // 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.visual_dim//16, 2, kernel_size=1, bias=False))

        self.seg_0 = nn.Sequential(
            ConvBNRelu(self.visual_dim, self.visual_dim // 4, kernel_size=3, stride=1, padding=1),
            ConvBNRelu(self.visual_dim // 4, self.visual_dim // 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.visual_dim//16, 2, kernel_size=1, bias=False))

        self.seg_1 = nn.Sequential(
            ConvBNRelu(self.visual_dim, self.visual_dim // 4, kernel_size=3, stride=1, padding=1),
            ConvBNRelu(self.visual_dim // 4, self.visual_dim // 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.visual_dim//16, 2, kernel_size=1, bias=False))

        self.seg_2 = nn.Sequential(
            ConvBNRelu(self.visual_dim, self.visual_dim // 4, kernel_size=3, stride=1, padding=1),
            ConvBNRelu(self.visual_dim // 4, self.visual_dim // 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.visual_dim//16, 2, kernel_size=1, bias=False))

        self.seg_3 = nn.Sequential(
            ConvBNRelu(self.visual_dim, self.visual_dim // 4, kernel_size=3, stride=1, padding=1),
            ConvBNRelu(self.visual_dim // 4, self.visual_dim // 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.visual_dim//16, 2, kernel_size=1, bias=False))

        self.apply(self._init_weights)

        # backbone
        self.text_backbone = BERTEncoder(weight=self.bert_weight)
        self.backbone = swin_backbone()

    # trunc_normal
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m,nn.Conv2d):
            nn.init.trunc_normal_(m.weight,std=.02)

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path,map_location='cuda')
        self.load_state_dict(state_dict)

    def load_backbone_states(self,path):
        # load pretrained weight of visual backbone

        Base_Backbone_Statedict = torch.load(path)["model"]
        for k in list(Base_Backbone_Statedict.keys()):
            if "head" in k:
                del Base_Backbone_Statedict[k]

        self.backbone.load_states(Base_Backbone_Statedict)


    def forward(self,img, word_id, attn_mask, pos_onehot,adj):

        img = self.img_bn(img)

        # word-level and sent-level feature
        words, context = self.text_backbone(word_id, attn_mask)

        # words_feature
        b_size, l_len, embed_dim = words.shape
        words = words.view(-1, embed_dim)
        words_map = F.normalize(self.mapping_word(words).view(b_size, l_len, -1),p=2,dim=2)

        # pos feature
        pos_feature = F.normalize(self.pos_encoder(pos_onehot), p=2, dim=2)

        # pos words feature fuse
        pos_aware_words = self.word_pos(words_map, pos_feature)

        # visual feature
        v_0,v_1,v_2,v_3 = self.backbone(img)

        ###v_0###
        # v_0 #global text feature
        coord = generate_coord(v_0.size(0),v_0.size(2),v_0.size(3))
        v_0_temp = F.normalize(v_0, p=2, dim=1)

        # text-global
        # sentence feature
        HT = F.normalize(self.mapping_hT(context),p=2,dim=1)
        HT_tile = HT.view(HT.size(0),HT.size(1),1,1).repeat(1,1,v_0.size(2),v_0.size(3))

        vt0 = self.v_0_conv2(torch.cat([v_0_temp, HT_tile, coord], dim=1))

        ###v_1###
        coord = generate_coord(v_1.size(0),v_1.size(2),v_1.size(3))
        v_1_temp = F.normalize(v_1, p=2, dim=1)
        v_1_cat = F.normalize(self.v_1_conv2(torch.cat([v_1_temp, coord], dim=1)),p=2,dim=1)
        v_t_1 = self.wordvisual1(pos_aware_words, v_1_cat)

        # text graph reasoning
        rel_aware_words_1 = self.text_graph_1(pos_aware_words,adj)
        # visual graph reasoning
        v_graph_1 = self.vis_graph_1(v_1_cat)
        # node weighted sum
        v_t_graph_1 = self.graph_wordvisual1(rel_aware_words_1, v_graph_1)
        # node fuse
        vt1 = self.fea_fuse_1(torch.cat([v_t_1,v_t_graph_1], dim=1))


        ###v_2###
        coord = generate_coord(v_2.size(0),v_2.size(2),v_2.size(3))
        v_2_temp = F.normalize(v_2, p=2, dim=1)
        v_2_cat = F.normalize(self.v_2_conv2(torch.cat([v_2_temp, coord],dim=1)),p=2,dim=1)
        v_t_2 = self.wordvisual2(pos_aware_words, v_2_cat)

        # text graph reasoning
        rel_aware_words_2 = self.text_graph_2(pos_aware_words,adj)
        # visual graph reasoning
        v_graph_2 = self.vis_graph_2(v_2_cat)
        # node weighted sum
        v_t_graph_2 = self.graph_wordvisual2(rel_aware_words_2, v_graph_2)
        # node fuse
        vt2 = self.fea_fuse_2(torch.cat([v_t_2,v_t_graph_2], dim=1))


        ###v_3###
        coord = generate_coord(v_3.size(0),v_3.size(2),v_3.size(3))
        v_3_temp = F.normalize(v_3, p=2, dim=1)
        v_3_cat = F.normalize(self.v_3_conv2(torch.cat([v_3_temp, coord],dim=1)),p=2,dim=1)
        v_t_3 = self.wordvisual3(pos_aware_words, v_3_cat)

        # text graph reasoning
        rel_aware_words_3 = self.text_graph_3(pos_aware_words,adj)
        # visual graph reasoning
        v_graph_3 = self.vis_graph_3(v_3_cat)
        # node weighted sum
        v_t_graph_3 = self.graph_wordvisual3(rel_aware_words_3, v_graph_3)
        # node fuse
        vt3 = self.fea_fuse_3(torch.cat([v_t_3,v_t_graph_3], dim=1))

        ########## decoder ###########
        vt3_final = F.normalize(vt3, p=2, dim=1)
        vt2_final = F.normalize(vt2, p=2, dim=1)
        vt1_final = F.normalize(vt1, p=2, dim=1)
        vt0_final = F.normalize(vt0, p=2, dim=1)

        # fuse multi-level feature map
        v_2_fuse = self.fuse_2(torch.cat([Upsample(vt3_final, vt2_final.shape[2:]), vt2_final], dim=1))
        v_1_fuse = self.fuse_1(torch.cat([Upsample(v_2_fuse, vt1_final.shape[2:]), vt1_final], dim=1))
        fuse = self.fuse_0(torch.cat([Upsample(v_1_fuse, vt0_final.shape[2:]), vt0_final], dim=1))

        # seg out
        seg_out = self.seg_out(fuse)
        seg_out = F.interpolate(seg_out,scale_factor=4, mode='bilinear', align_corners=False)

        # multi-scale
        seg_scale0 = self.seg_0(vt0)
        seg_scale1 = self.seg_1(vt1)
        seg_scale2 = self.seg_2(vt2)
        seg_scale3 = self.seg_3(vt3)

        return [seg_out, seg_scale0, seg_scale1, seg_scale2, seg_scale3]
