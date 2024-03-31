import numpy as np
import cv2
import os
import spacy
import torch
import transformers
from torch.utils.data import Dataset
import torch.nn.functional as F


class Generator(Dataset):

    """ Abstract generator class.
    """

    def __init__(
        self,
        data,
        config,
        train_mode=True,
    ):
        self.data = data
        self.config = config
        self.train_mode = train_mode
        self.embed = spacy.load(config.word_embed)
        self.input_shape = (config.input_size, config.input_size)


    def __getitem__(self, index):
        """
        torch sequence method for generating batches.
        """

        data = self.data[index]

        if self.train_mode:
            image, word_id, seg_map, att_mask, Adj_matrix, pos_onehot, mask_scale0, mask_scale1, mask_scale2, mask_scale3 = get_data(data,
                                                    self.input_shape,
                                                    self.embed,
                                                    self.config,
                                                    train_mode=self.train_mode)

            word_data = word_id
            image_data = image
            seg_data = seg_map
            att_mask = att_mask
            Adj_matrix = Adj_matrix
            pos_onehot = pos_onehot
            mask_scale0 = mask_scale0
            mask_scale1 = mask_scale1
            mask_scale2 = mask_scale2
            mask_scale3 = mask_scale3

            return [image_data, word_data, seg_data, att_mask, Adj_matrix, pos_onehot, mask_scale0, mask_scale1, mask_scale2, mask_scale3]

        if not self.train_mode:
             image_data, word_id, seg_map, ori_image, sentences, att_mask, Adj_matrix, pos_onehot , sent_len = get_data(data,
                                                       self.input_shape,
                                                       self.embed,
                                                       self.config,
                                                       train_mode=self.train_mode)
             word_data = word_id
             image_data = image_data
             ori_image = ori_image
             sentences = sentences
             seg_data = seg_map
             att_mask = att_mask
             Adj_matrix = Adj_matrix
             pos_onehot = pos_onehot
             sent_len = sent_len

             return [image_data, word_data, seg_data, ori_image, sentences, att_mask, Adj_matrix, pos_onehot, sent_len]

    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.data)


def qlist_to_id(max_length, q_list, tokenizer):
    '''
    note: 2018.10.3
    use for process sentences
    '''

    attention_mask = [0] * max_length
    padded_input_ids = [0] * max_length

    # `add_special_tokens=True`加入<SOS>和<EOS>
    input_ids = tokenizer.encode(text=q_list, add_special_tokens=True)

    # truncation of tokens
    input_ids = input_ids[:max_length]

    padded_input_ids[:len(input_ids)] = input_ids
    attention_mask[:len(input_ids)] = [1] * len(input_ids)

    padded_input_ids = torch.tensor(padded_input_ids)
    attention_mask = torch.tensor(attention_mask)

    return padded_input_ids,attention_mask


def get_data(ref, input_shape, embed, config, train_mode=True):
    '''preprocessing for real-time data augmentation'''

    SEG_DIR = config.seg_gt_path
    h, w = input_shape

   # print(ref)
    seg_id = ref['segment_id']

    sentences = ref['sentences']
    choose_index = 0 #every data has only 1 sentence

    sent = sentences[choose_index]['sent']

    tokenizer = transformers.BertTokenizer.from_pretrained(config.BERT_weight)
    # get adj and pos
    Adj_matrix, pos_onehot ,sent_len = sent_preprocess(embed, sent, config.word_len, tokenizer)
    # get word_id and att mask for BERT
    word_id ,att_mask = qlist_to_id(config.word_len, sent, tokenizer)

    #get img
    image = cv2.imread(os.path.join(config.image_path, ref['img_name']))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ih, iw, _ = image.shape

    if not train_mode:
        ori_image = image.copy()

    image_data = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)

    # get mask
    seg_map = cv2.imread(os.path.join(SEG_DIR, str(seg_id)+'.png'), flags=cv2.IMREAD_GRAYSCALE)

    if train_mode:

        seg_map_data = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST) / 255.0

        # multi scale
        seg_map_scale0 = cv2.resize(seg_map_data, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
        seg_map_scale1 = cv2.resize(seg_map_data, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST)
        seg_map_scale2 = cv2.resize(seg_map_data, (w // 16, h // 16), interpolation=cv2.INTER_NEAREST)
        seg_map_scale3 = cv2.resize(seg_map_data, (w // 32, h // 32), interpolation=cv2.INTER_NEAREST)

        seg_map_data = seg_map_data[:, :, None]
        seg_map_scale0 = seg_map_scale0[:, :, None]
        seg_map_scale1 = seg_map_scale1[:, :, None]
        seg_map_scale2 = seg_map_scale2[:, :, None]
        seg_map_scale3 = seg_map_scale3[:, :, None]

        # numpy to tensor
        seg_map_data = torch.from_numpy(seg_map_data).permute(2, 0, 1)
        seg_map_scale0 = torch.from_numpy(seg_map_scale0).permute(2, 0, 1)
        seg_map_scale1 = torch.from_numpy(seg_map_scale1).permute(2, 0, 1)
        seg_map_scale2 = torch.from_numpy(seg_map_scale2).permute(2, 0, 1)
        seg_map_scale3 = torch.from_numpy(seg_map_scale3).permute(2, 0, 1)

    # numpy to tensor
    image_data = torch.from_numpy(image_data).permute(2,0,1)

    if train_mode:
        return image_data, word_id, seg_map_data, att_mask, Adj_matrix, pos_onehot, seg_map_scale0, seg_map_scale1,seg_map_scale2,seg_map_scale3

    if not train_mode:
        # word_vec = [qlist_to_vec(config.word_len, sent['sent'], embed) for sent in sentences]
        return image_data, word_id, seg_map[:, :, None],  ori_image,  sentences, att_mask, Adj_matrix, pos_onehot ,sent_len


def sent_preprocess(nlp, sent, max_length, tokenizer):

    token = tokenizer.tokenize(sent)
    # delete # made by bert tokenizer
    sent = ' '.join(token).replace('#','')

    # parser
    depend = []
    doc = nlp(str(sent))  # doc表示当前的sentence
    sent_len = len(doc)

    for token in doc:
        depend.append((token.i, token.head.i)) # 将关系对加入depend[]

    # <SOS>和<EOS> set 0
    edge = np.zeros((max_length,max_length),dtype=int)  # 创建n*n的矩阵
    for (i, j) in depend:
        if i >= max_length-1  or j >= max_length-1:
            continue
        edge[i+1][j+1] = 1

    edge = torch.from_numpy(edge).long()
    # edge = edge.t()

    # pos
    # <SOS>和<EOS> set 0
    pos = []
    for token in doc:
        pos.append(token.pos_)
    pos_label = np.zeros((max_length),dtype=int)


    for i,label in enumerate(pos):
        if i >= max_length-1:
            break

        if label in ['NOUN','PROPN']:
            pos_label[i+1] = 1
        elif label in ['ADJ']:
            pos_label[i+1] = 2
        elif label in ['VERB']:
            pos_label[i+1] = 3
        elif label in ['ADP']:
            pos_label[i+1] = 4
        elif label in ['ADV']:
            pos_label[i+1] = 5
        else:
            # other wise
            pos_label[i+1] = 6

    pos_label = torch.from_numpy(pos_label)
    pos_onehot = F.one_hot(pos_label,num_classes=7)

    return edge,pos_onehot,sent_len