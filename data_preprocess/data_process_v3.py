# encoding=utf8

import numpy as np
import os
from refer import REFER
import cv2
import argparse
from tqdm import tqdm
import json
parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument('--data_root',  type=str)
parser.add_argument('--output_dir',  type=str)
parser.add_argument('--dataset', type=str, choices=['refcoco', 'refcoco+', 'refcocog', 'referit'], default='refcoco')
parser.add_argument('--split',  type=str, default='umd')
parser.add_argument('--generate_mask',  action='store_true')
args = parser.parse_args()

if args.dataset == 'referit':
    splits = ['trainval', 'test']
elif args.dataset == 'refcoco':
    splits = ['train', 'val', 'testA', 'testB']
elif args.dataset == 'refcoco+':
    splits = ['train', 'val', 'testA', 'testB']
elif args.dataset == 'refcocog':
    splits = ['train', 'val', 'test']  # we don't have test split for refcocog right now.

if args.dataset == "referit":

    # data directory
    im_dir = os.path.join(args.data_root,'./images/referit/images/')
    mask_dir = os.path.join(args.data_root, './images/referit/mask/')
    output_dir = args.output_dir

    split_idx = 0
    for split in splits:
        ann_path = os.path.join(output_dir, 'anns', args.dataset)
        mask_path = os.path.join(output_dir, 'masks', args.dataset)
        if not os.path.exists(ann_path):
            os.makedirs(ann_path)
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)
        query_file = './referit/referit_query_' + split + '.json'
        # load annotations
        query_dict = json.load(open(query_file))
        im_list = query_dict.keys()
        sentence_list = query_dict.values()
        ref_idx = np.linspace(split_idx, split_idx + len(query_dict) - 1, len(query_dict)).astype(int)

        dataset_array = []
        print('Processing split:{} - Len: {}'.format(split, np.alen(ref_idx)))

        for i in tqdm(ref_idx):
            # i is whole id
            i = int(i)
            ii = i - split_idx

            im = list(im_list)[ii]
            img_url, group = im.split("_")
            img_url = img_url + '.jpg'  # img_name

            # img_path = os.path.join(im_dir,img_url)
            # if not os.path.exists(img_path):
            #     print(os.path.exists(img_path))
            #     continue
            if img_url in ['19579.jpg', '17975.jpg', '19575.jpg']:#referit cannot open
                print("delete {}".format(img_url))
                print("mask id = {}".format(i))
                continue

            mask = im + '.mat'

            sentences = list(sentence_list)[ii]


            for j, _ in enumerate(sentences):
                ref_dict = {}
                sent_dict = []

                ref_dict['segment_id'] = str(i) + '_' + str(j)
                ref_dict['img_name'] = img_url


                sent = sentences

                sent_dict.append({
                    'idx': j,
                    'sent': sent[j].strip()})

                ref_dict['sentences'] = sent_dict
                ref_dict['sentences_num'] = len(sent_dict)

                dataset_array.append(ref_dict)

                from scipy.io import loadmat

                matmask_file_path = os.path.join(mask_dir, mask)
                mask_load = loadmat(matmask_file_path)['segimg_t'] * (-255.0)
                mask_load = 255.0 - mask_load

                if args.generate_mask:
                    cv2.imwrite(os.path.join(mask_path, str(i)+'_'+str(j) + '.png'), mask_load)

        print('Dumping json file...')
        with open(os.path.join(output_dir, 'anns', args.dataset, split + '.json'), 'w') as f:
            json.dump(dataset_array, f)

        split_idx = len(query_dict)

else:
    #refcoco refcoco+ refcocog
    #h, w = (416, 416)

    refer = REFER(args.data_root, args.dataset, args.split)

    print('dataset [%s_%s] contains: ' % (args.dataset, args.split))
    ref_ids = refer.getRefIds()
    image_ids = refer.getImgIds()
    print('%s expressions for %s refs in %s images.' % (len(refer.Sents), len(ref_ids), len(image_ids)))

    print('\nAmong them:')

    for split in splits:
        ref_ids = refer.getRefIds(split=split)
        print('%s refs are in split [%s].' % (len(ref_ids), split))


    def cat_process(cat):
        if cat >= 1 and cat <= 11:
            cat = cat - 1
        elif cat >= 13 and cat <= 25:
            cat = cat - 2
        elif cat >= 27 and cat <= 28:
            cat = cat - 3
        elif cat >= 31 and cat <= 44:
            cat = cat - 5
        elif cat >= 46 and cat <= 65:
            cat = cat - 6
        elif cat == 67:
            cat = cat - 7
        elif cat == 70:
            cat = cat - 9
        elif cat >= 72 and cat <= 82:
            cat = cat - 10
        elif cat >= 84 and cat <= 90:
            cat = cat - 11
        return cat


    def bbox_process(bbox):
        x_min = int(bbox[0])
        y_min = int(bbox[1])
        x_max = x_min + int(bbox[2])
        y_max = y_min + int(bbox[3])
        return list(map(int, [x_min, y_min, x_max, y_max]))

    #数据整合为dict
    def prepare_dataset(dataset, splits, output_dir, generate_mask=False):
        ann_path = os.path.join(output_dir, 'anns', dataset)
        mask_path = os.path.join(output_dir, 'masks', dataset)
        if not os.path.exists(ann_path):
            os.makedirs(ann_path)
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)

        for split in splits:

            dataset_array = []
            ref_ids = refer.getRefIds(split=split)
            print('Processing split:{} - Len: {}'.format(split, np.alen(ref_ids)))
            for i in tqdm(ref_ids):


                refs = refer.Refs[i]
                bboxs = refer.getRefBox(i)
                sentences = refs['sentences']
                image_urls = refer.loadImgs(image_ids=refs['image_id'])[0]

                cat = cat_process(refs['category_id'])
                image_urls = image_urls['file_name']

                if dataset == 'refclef' and image_urls in ['19579.jpg', '17975.jpg', '19575.jpg']:
                    continue
                box_info = bbox_process(bboxs)
                # dont need
                # ref_dict['bbox'] = box_info
                # ref_dict['cat'] = cat

                for j, sent in enumerate(sentences):
                    ref_dict = {}
                    sent_dict = []

                    ref_dict['segment_id'] = str(i)+'_'+str(j)
                    ref_dict['img_name'] = image_urls

                    sent_dict.append({
                        'idx': j,
                        'sent': sent['sent'].strip()})

                    ref_dict['sentences'] = sent_dict
                    ref_dict['sentences_num'] = len(sent_dict)

                    dataset_array.append(ref_dict)
                    # print(dataset_array)
                    if generate_mask:
                        cv2.imwrite(os.path.join(mask_path, str(i)+'_'+str(j) + '.png'), refer.getMask(refs)['mask'] * 255)

            print('Dumping json file...')
            with open(os.path.join(output_dir, 'anns', dataset, split + '.json'), 'w') as f:
                json.dump(dataset_array, f)

    prepare_dataset(args.dataset, splits, args.output_dir, args.generate_mask)
