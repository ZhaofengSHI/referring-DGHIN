import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
import torch
from loader.config_loader import load_config,load_data
import torch.nn as nn
import argparse
import torch.backends.cudnn as cudnn
from models.model import model
from models.utils import Loss
from utils.functions import MovingAverage, SavePath
import math
import time
import torch.optim as optim
from utils import timer
import datetime
# Oof
import eval as eval_script
from utils.poly_lr_decay import PolynomialLRDecay
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup

from torch.utils.data.distributed import DistributedSampler


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(
    description='Training Script')
parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')
parser.add_argument('--num_gpus', type=int, default=4, help='node rank for distributed training')
parser.add_argument('--batch_size', default=6, type=int,
                    help='Batch size for training')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--cfg_dir',default='./config/',type=str,help='path to cfg dir')
parser.add_argument('--dataset', type=str, choices=['refcoco', 'refcocop', 'refcocog_umd', 'refcocog_google', 'referit'], default='refcoco')
parser.add_argument('--validation_epoch', default=1, type=int,
                    help='Output val information every n iters. If -1, no valition')
parser.add_argument('--init_lr', default=3e-5, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--end_lr', default=3e-7, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=0.01, type=float,
                    help='Weight decay for AdamW. Leave as None to read this from the config.')
parser.add_argument('--power', '--poly_power', default=2.0, type=float,
                    help='power for poly lr sche.')
parser.add_argument('--freeze_bn', action='store_true',help='freeze_bn')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--backbone_weights',action='store_false',help='load backbone pretrain weight or not')
parser.add_argument('--save_folder', default='./results/',
                    help='Directory for saving checkpoint models.')
parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                    help='will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')
parser.add_argument('--max_iter', default=320000.0, type=float, help='max_iter')
parser.add_argument('--warmup_iter', default=40000.0, type=float, help='warmup_iter')
parser.add_argument('--start_iter', default=-1, type=int,
                    help='Resume training at this iter. If this is -1, the iteration will be'\
                         'determined from the file name.')
parser.add_argument('--save_interval', default=40000, type=int,
                    help='The number of iterations between saving the model.')
parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
                    help='Only keep the latest checkpoint instead of each one.')
parser.add_argument('--keep_latest_interval', default=100000, type=int,
                    help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                    help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
parser.add_argument('--validation_size', default=-1, type=int,
                    help='The number of images to use for validation.')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
                         ', the model will resume training from the interrupt file.')

args = parser.parse_args()

# Load config
config = load_config(args.cfg_dir, args.dataset)
print(config)
print('\n--------------------------')

# make dir to store results
if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

# autoscale settings
if args.autoscale and args.batch_size != 8:
    factor = args.batch_size / 8
    print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, args.batch_size))

    args.max_iter //= (factor*args.num_gpus)
    args.warmup_iter //= (factor*args.num_gpus)
    args.save_interval //= (factor*args.num_gpus)


def torch_seed(seed=666):

    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

torch_seed()

#set default tensor type
if torch.cuda.is_available():
    if args.cuda:
        device = f'cuda:{args.local_rank}'
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        device = "cpu"
else:
    device = "cpu"


class NetLoss(nn.Module):

    def __init__(self, net, criterion):
        super().__init__()

        self.net = net


        self.criterion = criterion

    def forward(self, images, word_id, masks, att_mask, pos, adj, scale0, scale1, scale2, scale3):

        gts = [masks, scale0, scale1, scale2, scale3]
        preds = self.net(images, word_id, att_mask ,pos, adj)
        losses = self.criterion(preds, gts)

        return losses

if args.batch_size < 4:
    print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
    args.freeze_bn = True

loss_types = ['seg_loss','scale0_loss','scale1_loss','scale2_loss','scale3_loss']


def train():

    # initialize devices
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )
    device = torch.device(f'cuda:{args.local_rank}')

    train_dataset, val_dataset = load_data(config)
    print('\n--------------------------')

    #set up eval
    setup_eval()

    # wraps the underlying module, but when saving and loading we don't want that
    # when loading or other operations on the baseline_Net, net are synchronized

    baseline_Net = model(config=config)
    net = baseline_Net
    net.train()

    # Load backbone pretrain weights
    if args.backbone_weights:
        backbone_weight_path = config.backbone_weight
        baseline_Net.load_backbone_states(backbone_weight_path)

    # I don't use the timer during training (I use a different timing method).
    timer.disable_all()

    # Both of these can set args.resume to None, so do them before the check
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, config.name)

    if args.resume is not None:
        print('Resuming training, loading {}...'.format(args.resume))
        baseline_Net.load_weights(args.resume)

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration

    # optim and criterion
    optimizer = optim.AdamW(net.parameters(), lr=args.init_lr, weight_decay=args.decay)
    criterion = Loss()

    if args.cuda:
        net = net.to(device)

    # Initialize everything (freeze_bn--random init net--open_bn)
    if not args.freeze_bn: baseline_Net.freeze_bn() # Freeze bn so we don't kill our means
    net(torch.zeros(args.batch_size, 3, config.input_size, config.input_size).cuda(),#img
        torch.zeros(args.batch_size, config.word_len).long().cuda(),#word_id
        torch.zeros(args.batch_size, config.word_len).float().cuda(),#att_mask
        torch.zeros(args.batch_size, config.word_len, 7).cuda(),#pos
        torch.ones(args.batch_size, config.word_len, config.word_len).cuda())#adj
    if not args.freeze_bn: baseline_Net.freeze_bn(True)

    # if args.freeze_bn freeze bn
    if args.freeze_bn:baseline_Net.freeze_bn(enable=False)

    # convert sync bn
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    # convert model to distributed model
    net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
    # Net loss counter
    net = NetLoss(net, criterion)

    # loss counters
    iteration = max(args.start_iter, 0) #initializing iteration
    last_time = time.time()

    ######
    epoch_size = len(train_dataset) // (args.batch_size * args.num_gpus)
    num_epochs = math.ceil(args.max_iter / epoch_size)
    num_train_epochs = math.ceil((args.max_iter - args.warmup_iter) / epoch_size) + 1

    # poly lr scheduler
    scheduler = PolynomialLRDecay(optimizer,
                                 max_decay_steps=num_train_epochs,
                                 end_learning_rate=args.end_lr,
                                 power=args.power)

    lr_scheduler_warmup = create_lr_scheduler_with_warmup(lr_scheduler=scheduler,
                                                          warmup_start_value=1e-10,
                                                          warmup_duration=int(args.warmup_iter),
                                                          warmup_end_value=args.init_lr)

    # data_loader
    Train_Dataloader = torch.utils.data.DataLoader(train_dataset,sampler=DistributedSampler(train_dataset,shuffle=True),
                                                   batch_size=args.batch_size,shuffle=False, num_workers=args.num_workers,pin_memory=True)

    save_path = lambda epoch, iteration: SavePath(config.name, epoch, iteration).get_path(root=args.save_folder)
    time_avg = MovingAverage()

    global loss_types # Forms the print order
    loss_avgs = { k: MovingAverage(100) for k in loss_types }

    print('Begin training!')
    print('\n--------------------------')
    # try-except so you can use ctrl+c to save early and stop training
    try:
        for epoch in range(num_epochs):
            # set epoch
            Train_Dataloader.sampler.set_epoch(epoch)
            # Resume from start_iter
            if (epoch+1)*epoch_size < iteration:
                if iteration > args.warmup_iter:
                    # adjust lr when end of epoch
                    scheduler.step()

                continue

            for data in Train_Dataloader:

                # Stop if we've reached an epoch if we're resuming from start_iter
                if iteration == (epoch+1)*epoch_size:
                    break

                # Stop at the configured number of iterations even if mid-epoch
                if iteration == args.max_iter:
                    break

                # Zero the grad to get ready to compute gradients
                optimizer.zero_grad()

                # lr warmup
                if iteration <= args.warmup_iter:
                    lr_scheduler_warmup(None)

                # get img/sent/mask/sent_len data
                imgs = data[0].float().to(device)
                word_id = data[1].long().to(device)
                masks = data[2].squeeze(1).long().to(device)
                att_mask = data[3].float().to(device)
                Adj_matrix = data[4].long().to(device)
                pos_onehot = data[5].float().to(device)

                # multi scale masks
                mask_scale0 = data[6].squeeze(1).long().to(device)
                mask_scale1 = data[7].squeeze(1).long().to(device)
                mask_scale2 = data[8].squeeze(1).long().to(device)
                mask_scale3 = data[9].squeeze(1).long().to(device)

                # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
                losses = net(images = imgs, word_id = word_id, masks = masks,att_mask = att_mask,pos = pos_onehot,adj = Adj_matrix,
                             scale0 = mask_scale0, scale1= mask_scale1,scale2= mask_scale2,scale3= mask_scale3)
                loss = sum([losses[k] for k in losses]) # add diff loss

                # Backprop
                loss.backward()

                if torch.isfinite(loss).item():# Do this to free up vram even if loss is not finite
                    optimizer.step()

                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    loss_avgs[k].add(losses[k].item()) #Keeps an moving average window of losses

                cur_time = time.time()
                elapsed = cur_time - last_time
                last_time = cur_time

                # Exclude graph setup from the timing information
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                if iteration % 10 == 0:

                    # print present lr
                    lr = optimizer.param_groups[0]['lr']

                    # print and load losses and time
                    eta_str = str(datetime.timedelta(seconds=(args.max_iter - iteration) * time_avg.get_avg())).split('.')[0]

                    total = sum([loss_avgs[k].get_avg() for k in losses])
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])

                    if torch.distributed.get_rank() == 0:  ###

                        print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f || lr: %.8f')
                              % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed,lr]), flush=True)
                        logfile = open("logfile.txt", 'a')
                        print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f || lr: %.8f')
                              % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed,lr]), flush=True,
                              file=logfile)
                        logfile.close()

                iteration += 1

                if iteration % args.save_interval == 0 and iteration != args.start_iter:

                    if args.keep_latest:
                        latest = SavePath.get_latest(args.save_folder, config.name)

                    print('Saving state, iter:', iteration)

                    if torch.distributed.get_rank() == 0:###
                        baseline_Net.save_weights(save_path(epoch, iteration))
                        compute_validation_map(epoch, iteration, baseline_Net, val_dataset)

                    if args.keep_latest and latest is not None:
                        if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                            print('Deleting old save...')
                            os.remove(latest)

            if iteration > args.warmup_iter:
                # adjust lr when end of epoch
                scheduler.step()


        # Compute validation mAP after training is finished
        compute_validation_map(epoch, iteration, baseline_Net, val_dataset)

    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        if args.interrupt:
            print('Stopping early. Saving network...')
            if torch.distributed.get_rank() == 0:  ###
                # Delete previous copy of the interrupted network so we don't spam the weights folder
                SavePath.remove_interrupt(args.save_folder)
            baseline_Net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))
        exit()


#validation
def compute_validation_map(epoch, iteration, net, dataset):
    with torch.no_grad():

        net.eval()
        # start = time.time()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info = eval_script.evaluate(net, dataset)
        # end = time.time()
        logfile = open("logfile.txt", 'a')
        print('iter:{} epoch:{}: {}'.format(int(iteration / args.save_interval), epoch, val_info), file=logfile)
        logfile.close()
        logfile = open("acc.txt", 'a')
        print('iter:{} epoch:{}: {}'.format(int(iteration / args.save_interval), epoch, val_info), file=logfile)
        logfile.close()
        # val_info = {"val_info": val_info}

        net.train()

def setup_eval():
    eval_script.parse_args(['--no_bar', '--max_images='+str(args.validation_size)])

if __name__ == '__main__':
    train()





