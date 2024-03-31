import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from loader.config_loader import load_config, load_data
from utils import timer
import numpy as np
import torch
from utils.functions import SavePath
from torch.autograd import Variable
import argparse
import random
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from models.model import model
import time

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Evaluation')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--cfg_dir',default='./config/',type=str,help='path to cfg dir')
    parser.add_argument('--dataset', type=str, choices=['refcoco', 'refcoco+', 'refcocog', 'referit'], default='refcoco')
    parser.add_argument('--trained_model',
                        default='latest', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')

    global args
    args = parser.parse_args(argv)

    return args



def predmask_resize(gt_w, gt_h, pred_mask):

    pred_resize = F.interpolate(pred_mask, (gt_w, gt_h), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

    return pred_resize

# all boxes are [num, height, width] binary array
def compute_mask_IU(masks, target):
    assert (target.shape[-2:] == masks.shape[-2:])
    I = np.sum(np.logical_and(masks, target))
    U = np.sum(np.logical_or(masks, target))
    return I, U

def evaluate(net, dataset):


    if args.cuda:
        cudnn.fastest = True
        device = "cuda"

    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        device = "cpu"


    dataset_size = len(dataset) if args.max_images < 0 else min(args.max_images, len(dataset))
    dataset_indices = list(range(len(dataset)))


    if args.shuffle:
        random.shuffle(dataset_indices)

    dataset_indices = dataset_indices[:dataset_size]
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    cum_I, cum_U = 0, 0
    seg_total = 0.

    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    t_prediction = 0.0

    #warm up net
    net(torch.zeros(1, 3, 416, 416).cuda(), torch.zeros(1, 17).long().cuda(), torch.zeros(1, 17).cuda(),torch.zeros(1, 17, 7).cuda(),torch.ones(1, 17, 17).cuda())
    net(torch.zeros(1, 3, 416, 416).cuda(), torch.zeros(1, 17).long().cuda(), torch.zeros(1, 17).cuda(),torch.zeros(1, 17, 7).cuda(), torch.ones(1, 17, 17).cuda())
    net(torch.zeros(1, 3, 416, 416).cuda(), torch.zeros(1, 17).long().cuda(), torch.zeros(1, 17).cuda(),torch.zeros(1, 17, 7).cuda(), torch.ones(1, 17, 17).cuda())
    net(torch.zeros(1, 3, 416, 416).cuda(), torch.zeros(1, 17).long().cuda(), torch.zeros(1, 17).cuda(),torch.zeros(1, 17, 7).cuda(), torch.ones(1, 17, 17).cuda())
    net(torch.zeros(1, 3, 416, 416).cuda(), torch.zeros(1, 17).long().cuda(), torch.zeros(1, 17).cuda(),torch.zeros(1, 17, 7).cuda(), torch.ones(1, 17, 17).cuda())

    try:
        # eval loop
        for iter, img_idx in enumerate(dataset_indices):



            timer.reset()

            data_item = dataset.__getitem__(img_idx)
            img = data_item[0]
            word_ids = data_item[1]
            seg_data = data_item[2]
            ori_image = data_item[3]
            sentence = data_item[4]
            att_mask = data_item[5]
            Adj_matrix = data_item[6]
            pos_onehot = data_item[7]

            #type transfer

            pos_onehot = pos_onehot
            Adj_matrix = Adj_matrix

            # add batch dim
            img_batch = Variable(img.unsqueeze(0)).to(device).float()
            word_batch = Variable(word_ids.unsqueeze(0)).to(device).long()
            att_batch = Variable(att_mask.unsqueeze(0)).to(device).float()
            pos_onehot = Variable(pos_onehot.unsqueeze(0)).to(device).float()
            Adj_matrix = Variable(Adj_matrix.unsqueeze(0)).to(device).long()


            with timer.env('Network Extra'):
                start_time = time.time()

                preds = net(img_batch,word_batch,att_batch,pos_onehot,Adj_matrix)

                diff_time = time.time() - start_time
                t_prediction += diff_time
                print('Detection took {}s per image'.format(diff_time))

            # get [0-1] mask
            preds = preds[0].argmax(1).unsqueeze(1).float()

            gt_w = seg_data.shape[0]
            gt_h = seg_data.shape[1]

            pred_mask = predmask_resize(gt_w,gt_h,preds)

            seg_data = np.squeeze(seg_data,axis=2)

            I, U = compute_mask_IU(seg_data, pred_mask.detach().cpu().numpy())
            cum_I += I
            cum_U += U

            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (I / U >= eval_seg_iou)
            seg_total += 1

        # Print results
        print('\nSegmentation evaluation (without DenseCRF):')
        result_str = ''
        for n_eval_iou in range(len(eval_seg_iou_list)):
            result_str += 'precision@%s = %f\n' % \
                          (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] / seg_total)
        result_str += 'overall IoU = %f\n' % (cum_I / cum_U)
        print(result_str)

        print("Total prediction time: {}. Average: {}/image. {}FPS".format(
            t_prediction, t_prediction / len(dataset_indices), 1.0 / (t_prediction / len(dataset_indices))))

    except KeyboardInterrupt:
        print('Stopping...')

    return result_str



if __name__ == '__main__':

    parse_args()

    print(args.cuda)
    if args.cfg_dir is not None:
        # Load config
        config = load_config(args.cfg_dir, args.dataset)
        print(config)
        print('\n--------------------------')

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('results/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('results/', config.name)

    with torch.no_grad():

        # load val dataset
        val_dataset = load_data(config,train_mode=False)

        # load model and weights
        print('Loading model...', end='')
        net = model(config=config)
        if args.cuda:
            net = net.cuda()
        print(args.trained_model)
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')



        val_info = evaluate(net, val_dataset)

        # output val result
        logfile = open("acc_val.txt", 'a')
        print('{}'.format(val_info), file=logfile)
        logfile.close()
