import argparse
import os
import random
import time
import shutil
from collections import OrderedDict

import matplotlib
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from scipy import stats
from torch.optim import lr_scheduler
from tqdm import tqdm

from dataloader.rgb1x1_traj_dataloader import train_data_loader, test_data_loader
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime

from losses.BMC import BMCLoss
from models.Quality_Model import Quality_Model
from utils.utils import save_file_setup, get_max_len_128, config_setup
from utils.vis_tools import RecorderMeter, AverageMeter, ProgressMeter


def setup_seed(args):
    """
    Set random seed for torch and numpy.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def lr_decay(optimizer, lr_now, gamma):
#     lr_new = lr_now * gamma
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr_new
#     return lr_new

# def BCE_loss(pred, label):
#     bce_label = torch.where(label>=1, torch.tensor(1.0), torch.tensor(0.0))
#     loss_bce = nn.BCEWithLogitsLoss().cuda()
#     loss = loss_bce(pred, bce_label)
#     return loss

def main(args):
    # print configuration
    line = '=' * 40 + '\n'
    output = line
    for k, v in vars(args).items():
        output += f'{k}: {v}\n'
    output += line
    print(output)

    with open(log_txt_path, 'a') as f:
        f.write(output)


    best_pho = -1
    best_epoch = 0

    max_len = get_max_len_128(args.class_idx)[0]
    # (N M), (K, L)
    t_partition_size = get_max_len_128(args.class_idx)[1]
    p_partition_size = get_max_len_128(args.class_idx)[2]

    recorder = RecorderMeter(args.epochs)
    print('The training time: ' + now.strftime("%m-%d %H:%M"))
    with open(log_txt_path, 'a') as f:
        f.write('The training time: ' + now.strftime("%m-%d %H:%M") + '\n')

    # create model and load pre_trained parameters

    model = Quality_Model(args=args,
                              in_channels=5, depths=(2,2,2,2), channels=(128,256,256,256), num_points=args.n_landmark, num_frames=max_len,
                 type_1_size=(t_partition_size[0], p_partition_size[1]),
                type_2_size=(t_partition_size[0], p_partition_size[0]),
                type_3_size=(t_partition_size[1], p_partition_size[1]),
                type_4_size=(t_partition_size[1], p_partition_size[0]),
                attn_drop=0.5, drop=0., rel=True, drop_path=0.2, mlp_ratio=2.,
                act_layer=nn.GELU, norm_layer_transformer=nn.LayerNorm
                          ).cuda()
        
    model.load_pretrained(args.ckpt_path)


    if len(args.gpu.split(',')) > 1:
        model = nn.DataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = BMCLoss(init_noise_sigma=args.noise_sigma) # for regression
    # criterion = nn.MSELoss().cuda()

    if args.optimizer =='SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.base_lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            weight_decay=args.weight_decay)
    elif args.optimizer =='Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.base_lr)
            # weight_decay=args.weight_decay)
    else:
        raise ValueError

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_gamma)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_pho = checkpoint['best_pho']
            recorder = checkpoint['recorder']
            best_pho = best_pho.cuda()
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # Data loading code
    train_data = train_data_loader(args)
    test_data = test_data_loader(args)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    

    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        print(inf)
        print('Current learning rate: ', current_learning_rate)

        # train for one epoch
        train_pho, train_los = train(train_loader, model, criterion, optimizer, scheduler, epoch, args)
        # evaluate on validation set
        val_pho, val_los = validate(val_loader, model, criterion, args)

        # remember best acc and save checkpoint
        is_best = val_pho > best_pho
        best_pho = max(val_pho, best_pho)
        if is_best:
            best_epoch = epoch

        save_checkpoint({'epoch': epoch + 1,
                         'model': model.state_dict(),
                         'best_pho': best_pho,
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'recorder': recorder}, is_best)

        # print and save log
        epoch_time = time.time() - start_time

        recorder.update(epoch, train_los, train_pho, val_los, val_pho)
        recorder.plot_curve(log_curve_path)

        print('The best rho: {:.5f} in epoch {}'.format(best_pho, best_epoch))
        print('An epoch time: {:.1f}s'.format(epoch_time))
        with open(log_txt_path, 'a') as f:
            f.write('The best rho: {:.5f}' + str(best_pho) + 'in {}'.format(best_epoch) + '\n')
            f.write('An epoch time: {:.1f}s' + str(epoch_time) + '\n')


def train(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    losses = AverageMeter('Loss', ':.4f')

    progress = ProgressMeter(len(train_loader),
                             [losses],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    true_scores = []
    pred_scores = []

    for idx, (data, index_t) in enumerate(train_loader):

        rgb = data['video'].cuda()
        trajs = data['traj'].float().cuda()  # N, T, 2
        label = data['final_score'].float().reshape(-1, 1).cuda()
        true_scores.extend(data['final_score'].numpy())

        index_t = index_t.float().cuda() #(1,128)

        pred_label, traj_feats, tempo_feats = model(rgb, trajs, index_t)

        pred_scores.extend([i.item() for i in pred_label])

        # print(label.shape, pred_label.shape)
        loss = criterion(label, pred_label)
        losses.update(loss.item(), trajs.size(0))

        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

       # print loss and pho
        if idx % args.print_freq == 0 or idx == len(train_loader)-1:
            progress.display(idx, log_txt_path)

    scheduler.step()

    # analysis on results
    pred_scores = np.array(pred_scores)
    true_scores = np.array(true_scores)
    rho, p = stats.spearmanr(pred_scores, true_scores)
    print('[train] EPOCH: %d,  correlation_v: %.4f,  lr: %.4f'
          % (epoch,  rho,  optimizer.param_groups[0]['lr']))
    if epoch == 2 or epoch == args.epochs-1:
        print('pred_scores', pred_scores)
        print('true_scores', true_scores)

    with open(log_txt_path, 'a') as f:
        f.write('[train] EPOCH: {epoch:d}'.format(epoch=epoch)+
                'correlation_v: {rho:.6f}'.format(rho=rho)+
                'current LR: {LR:.6f}'.format(LR=optimizer.param_groups[0]['lr']) + '\n')

    return rho, losses.avg


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(len(val_loader),
                             [losses],
                             prefix='Test: ')
    

    # switch to evaluate mode
    model.eval()
    true_scores = []
    pred_scores = []

    with torch.no_grad():
        for idx, (data, index_t) in enumerate(val_loader):
                rgb = data['video'].cuda()
                trajs = data['traj'].float().cuda()  # N, T, 2
                label = data['final_score'].float().reshape(-1, 1).cuda()
                true_scores.extend(data['final_score'].numpy())
                index_t = index_t.float().cuda()

                pred_label, traj_feats, tempo_feats = model(rgb, trajs, index_t)
                loss = criterion(label, pred_label)

                losses.update(loss.item(), trajs.size(0))

                pred_scores.extend([i.item() for i in pred_label])

                if idx % args.print_freq == 0 or idx == len(val_loader)-1:
                    progress.display(idx, log_txt_path)

        rho, p = stats.spearmanr(pred_scores, true_scores)

        # TODO: this should also be done with the ProgressMeter
        print('Current Pho: {rho:.6f}'.format(rho=rho))
        print('Predicted scores: ', pred_scores)
        print('True scores: ', true_scores)
        with open(log_txt_path, 'a') as f:
            f.write('Current Pho:  {rho:.6f}'.format(rho=rho) + '\n')
    return rho, losses.avg

def save_checkpoint(state, is_best):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #--------general options----------
    parser.add_argument('--data_root', type=str,
                        default='./dataset/PD/insightface_5_crop_41',
                        help='aug_test path')
    parser.add_argument('--ckpt_path', type=str,
                        default='./models/pretrained_weights/')
    parser.add_argument('--config_name', type=str, help='path to save tensorboard curve', default='RGB_SkateFormer_256_DFEW_rgb1x1')
    parser.add_argument('--seed', type=int, help='manual seed', default=-1)
    parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument('--benchmark', type=str, default='PD')#PD
    #----------dataloader parameters-----------
    parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='N')
    parser.add_argument('--k', type=int, default=5, help='num of augmentation chose')
    parser.add_argument('--class_idx', type=int, default=0, choices=[0, 1, 2, 3, 4], help='class idx in PD-5')
    parser.add_argument('--clip_len', type=int, default=80, help='input length')
    parser.add_argument('--n_landmark', type=int, default=63, help='number of landmarks ')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of aug_test loading workers')
    #----------feature extractor---------------
    parser.add_argument('--backbone', type=str, default='ST')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate in transformer')
    #----------loss parapmeters----------------
    parser.add_argument('--noise_sigma', default=1.0, type=float, help='parameter of BMC loss')
    #----------training parameters--------------
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-p', '--print_freq', default=20, type=int, metavar='N', help='print frequency')
    # optimizer
    parser.add_argument('--optimizer', default='Adam', type=str, choices=['Adam', 'SGD'])
    parser.add_argument('--decay_step', default=40, type=int)
    parser.add_argument('--decay_gamma', default=0.1, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    # Adam
    parser.add_argument('--base_lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    # SGD
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--nesterov', default=False, type=bool)

    parser.add_argument('--data_augment', default='1', type=str)
    parser.add_argument('--partition', default=True, type=bool)
    parser.add_argument('--pad_type', type=str, default='loop')
    parser.add_argument('--rgb_pad', type=str, default='loop')
    parser.add_argument('--long_pad_traj', type=str, default='latter_truncate')
    parser.add_argument('--traj_input_dim', type=int,default=5)
    parser.add_argument('--traj_len', type=int, default=128)

    args = parser.parse_args()
    exp_name = args.config_name + args.data_augment+'_traj_len'+str(args.traj_len) +'_traj_input_dim'+str(args.traj_input_dim) + '_batch_'+str(args.batch_size) + '_par_type_' + str(args.partition_type) +'_seed'+str(args.seed) +  '_trajpad'+str(args.pad_type) +  '_long_traj_pad' +str(args.long_pad_traj)
    config_setup(args)
    now = datetime.datetime.now()
    log_txt_path, log_curve_path, checkpoint_path, best_checkpoint_path = save_file_setup(args, exp_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main(args)

