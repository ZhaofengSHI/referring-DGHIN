# encoding=utf8

import numpy as np
import os
from refer import REFER
import cv2
import argparse
from tqdm import tqdm
import json
parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument('--data_root', default='.', type=str)
parser.add_argument('--output_dir', default='data_v6', type=str)

args = parser.parse_args()


#数据整合为dict
def prepare_dataset(dataset_item, splits, output_dir, generate_mask=True):

    ann_path = os.path.join(output_dir, 'anns', 'merge_refcoco+')
    mask_path = os.path.join(output_dir, 'masks', 'merge_refcoco+')

    if not os.path.exists(ann_path):
        os.makedirs(ann_path)
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    for split in splits:

        ref_ids = refer.getRefIds(split=split)
        print('Processing split:{} - Len: {}'.format(split, np.alen(ref_ids)))

        for i in tqdm(ref_ids):

            refs = refer.Refs[i]
            sentences = refs['sentences']
            image_urls = refer.loadImgs(image_ids=refs['image_id'])[0]
            image_urls = image_urls['file_name']

            # if dataset_item != 'refcocog':####
            if image_urls in all_test_imgs:
                print('remove:',image_urls)
                continue


            for j, sent in enumerate(sentences):
                ref_dict = {}
                sent_dict = []
                ref_dict['segment_id'] = str(dataset_item) + '_' + str(i)+'_'+str(j)

                ref_dict['img_name'] = image_urls

                sent_dict.append({
                    'idx': j,
                    'sent': sent['sent'].strip()})

                ref_dict['sentences'] = sent_dict
                ref_dict['sentences_num'] = len(sent_dict)

                dataset_array.append(ref_dict)

                if generate_mask:
                    cv2.imwrite(os.path.join(mask_path, str(dataset_item) + '_' + str(i)+'_'+str(j) + '.png'), refer.getMask(refs)['mask'] * 255)



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

if __name__ == '__main__':

    root_dir = './all_test_set_anno'
    datasets = ['refcoco+']#, 'refcoco+', 'refcocog_google']#
    all_test_imgs = []

    for i in range(len(datasets)):

        dataset_item = datasets[i]
        dataset_path = os.path.join(root_dir, dataset_item)

        anno_jsons = os.listdir(dataset_path)

        for j in range(len(anno_jsons)):

            anno_json_item = anno_jsons[j]
            anno_json_path = os.path.join(root_dir, dataset_item, anno_json_item)

            with open(anno_json_path, encoding='utf-8') as f:
                json_lists = json.load(f)

            for k in range(len(json_lists)):
                json_list = json_lists[k]
                img_name = json_list['img_name']

                all_test_imgs.append(img_name)

    print(len(all_test_imgs))


    dataset = ['refcoco', 'refcoco+', 'refcocog']
    partition = ['unc', 'unc', 'google']
    splits = ['train']

    dataset_array = []

    for m in range(len(dataset)):

        dataset_item = dataset[m]
        partition_item = partition[m]

        refer = REFER(args.data_root, dataset_item, partition_item)

        print('dataset [%s_%s] contains: ' % (dataset_item,partition_item))
        ref_ids = refer.getRefIds()
        image_ids = refer.getImgIds()
        print('%s expressions for %s refs in %s images.' % (len(refer.Sents), len(ref_ids), len(image_ids)))

        print('\nAmong them:')


        for split in splits:
            ref_ids = refer.getRefIds(split=split)
            print('%s refs are in split [%s].' % (len(ref_ids), split))

        prepare_dataset(dataset_item, splits, args.output_dir)


    print('Dumping json file...')
    with open(os.path.join(args.output_dir, 'anns', 'merge_refcoco+', 'train_merge' + '.json'), 'w') as f:#
        json.dump(dataset_array, f)






