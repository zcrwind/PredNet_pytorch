# -*- coding: utf-8 -*-


import os
import numpy as np
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler

# zcr lib
from prednet import PredNet
from data_utils import ZcrDataLoader

# os.environ['CUDA_LAUNCH_BLOCKING'] = 1
# torch.backends.cudnn.benchmark = True

def arg_parse():
    desc = "Video Frames Predicting Task via PredNet."
    parser = argparse.ArgumentParser(description = desc)

    parser.add_argument('--mode', default = 'train', type = str,
                        help = 'train or evaluate (default: train)')
    parser.add_argument('--dataPath', default = '', type = str, metavar = 'PATH',
                        help = 'path to video dataset (default: none)')
    parser.add_argument('--checkpoint_savePath', default = '', type = str, metavar = 'PATH',
                        help = 'path for saving checkpoint file (default: none)')
    parser.add_argument('--epochs', default = 20, type = int, metavar='N',
                        help = 'number of total epochs to run')
    parser.add_argument('--batch_size', default = 32, type = int, metavar = 'N',
                        help = 'The size of batch')
    parser.add_argument('--optimizer', default = 'SGD', type = str,
                        help = 'which optimizer to use')
    parser.add_argument('--lr', default = 0.01, type = float,
                        metavar = 'LR', help = 'initial learning rate')
    parser.add_argument('--momentum', default = 0.9, type = float,
                        help = 'momentum for SGD')
    parser.add_argument('--beta1', default = 0.9, type = float,
                        help = 'beta1 in Adam optimizer')
    parser.add_argument('--beta2', default = 0.99, type = float,
                        help = 'beta2 in Adam optimizer')
    parser.add_argument('--workers', default = 4, type = int, metavar = 'N',
                        help = 'number of data loading workers (default: 4)')
    parser.add_argument('--checkpoint_file', default = '', type = str,
                        help = 'path to checkpoint file for restrating (default: none)')
    parser.add_argument('--printCircle', default = 100, type = int, metavar = 'N',
                        help = 'how many steps to print the loss information')
    parser.add_argument('--data_format', default = 'channels_last', type = str,
                        help = '(c, h, w) or (h, w, c)?')
    parser.add_argument('--n_channels', default = 3, type = int, metavar = 'N',
                        help = 'The number of input channels (default: 3)')
    parser.add_argument('--img_height', default = 128, type = int, metavar = 'N',
                        help = 'The height of input frame (default: 128)')
    parser.add_argument('--img_width', default = 160, type = int, metavar = 'N',
                        help = 'The width of input frame (default: 160)')
    # parser.add_argument('--stack_sizes', default = '', type = str,
    #                     help = 'Number of channels in targets (A) and predictions (Ahat) in each layer of the architecture.')
    # parser.add_argument('--R_stack_sizes', default = '', type = str,
    #                     help = 'Number of channels in the representation (R) modules.')
    # parser.add_argument('--A_filter_sizes', default = '', type = str,
    #                     help = 'Filter sizes for the target (A) modules. (except the target (A) in lowest layer (i.e., input image))')
    # parser.add_argument('--Ahat_filter_sizes', default = '', type = str,
    #                     help = 'Filter sizes for the prediction (Ahat) modules.')
    # parser.add_argument('--R_filter_sizes', default = '', type = str,
    #                     help = 'Filter sizes for the representation (R) modules.')
    parser.add_argument('--layer_loss_weightsMode', default = 'L_0', type = str,
                        help = 'L_0 or L_all for loss weights in PredNet')
    parser.add_argument('--num_timeSteps', default = 10, type = int, metavar = 'N',
                        help = 'number of timesteps used for sequences in training (default: 10)')
    parser.add_argument('--shuffle', default = True, type = bool,
                        help = 'shuffle or not')
    
    args = parser.parse_args()
    return args

def print_args(args):
    print('-' * 50)
    for arg, content in args.__dict__.items():
        print("{}: {}".format(arg, content))
    print('-' * 50)

def train(model, args):
    '''Train PredNet on KITTI sequences'''
    
    # print('layer_loss_weightsMode: ', args.layer_loss_weightsMode)
    prednet = model
    # frame data files
    DATA_DIR = args.dataPath
    train_file = os.path.join(DATA_DIR, 'X_train.h5')
    train_sources = os.path.join(DATA_DIR, 'sources_train.h5')
    val_file = os.path.join(DATA_DIR, 'X_val.h5')
    val_sources = os.path.join(DATA_DIR, 'sources_val.h5')

    output_mode = 'error'
    sequence_start_mode = 'all'
    N_seq = None
    dataLoader = ZcrDataLoader(train_file, train_sources, output_mode, sequence_start_mode, N_seq, args).dataLoader()

    if prednet.data_format == 'channels_first':
        input_shape = (args.batch_size, args.num_timeSteps, n_channels, img_height, img_width)
    else:
        input_shape = (args.batch_size, args.num_timeSteps, img_height, img_width, n_channels)

    optimizer = torch.optim.Adam(prednet.parameters(), lr = args.lr)
    lr_maker  = lr_scheduler.StepLR(optimizer = optimizer, step_size = 75, gamma = 0.1)  # decay the lr every 50 epochs by a factor of 0.1

    printCircle = args.printCircle
    for e in range(args.epochs):
        tr_loss = 0.0
        sum_trainLoss_in_epoch = 0.0
        min_trainLoss_in_epoch = float('inf')
        startTime_epoch = time.time()
        lr_maker.step()

        initial_states = prednet.get_initial_states(input_shape)    # 原网络貌似不是stateful的, 故这里再每个epoch开始时重新初始化(如果是stateful的, 则只在全部的epoch开始时初始化一次)
        states = initial_states
        for step, (frameGroup, target) in enumerate(dataLoader):
            # print(frameGroup)   # [torch.FloatTensor of size 16x12x80x80]
            batch_frames = Variable(frameGroup.cuda())
            batch_y = Variable(target.cuda())
            output = prednet(batch_frames, states)

            # '''进行按照timestep和layer对error进行加权.'''
            ## 1. 按layer加权(巧妙利用广播. NOTE: 这里的error列表里的每个元素是Variable类型的矩阵, 需要转成numpy矩阵类型才可以用切片.)
            num_layer = len(stack_sizes)
            # weighting for each layer in final loss
            if args.layer_loss_weightsMode == 'L_0':        # e.g., [1., 0., 0., 0.]
                layer_weights = np.array([0. for _ in range(num_layer)])
                layer_weights[0] = 1.
                layer_weights = torch.from_numpy(layer_weights)
                # layer_weights = torch.from_numpy(np.array([1., 0., 0., 0.]))
            elif args.layer_loss_weightsMode == 'L_all':    # e.g., [1., 1., 1., 1.]
                layer_weights = np.array([0.1 for _ in range(num_layer)])
                layer_weights[0] = 1.
                layer_weights = torch.from_numpy(layer_weights)
                # layer_weights = torch.from_numpy(np.array([1., 0.1, 0.1, 0.1]))
            else:
                raise(RuntimeError('Unknown loss weighting mode! Please use `L_0` or `L_all`.'))
            # layer_weights = Variable(layer_weights.float().cuda(), requires_grad = False)  # NOTE: layer_weights默认是DoubleTensor, 而下面的error是FloatTensor的Variable, 如果直接相乘会报错!
            layer_weights = Variable(layer_weights.float().cuda())  # NOTE: layer_weights默认是DoubleTensor, 而下面的error是FloatTensor的Variable, 如果直接相乘会报错!
            error_list = [batch_x_numLayer__error * layer_weights for batch_x_numLayer__error in output]    # 利用广播实现加权

            ## 2. 按timestep进行加权. (paper: equally weight all timesteps except the first)
            num_timeSteps = args.num_timeSteps
            time_loss_weight  = (1. / (num_timeSteps - 1))
            time_loss_weight  = Variable(torch.from_numpy(np.array([time_loss_weight])).float().cuda())
            time_loss_weights = [time_loss_weight for _ in range(num_timeSteps - 1)]
            time_loss_weights.insert(0, Variable(torch.from_numpy(np.array([0.])).float().cuda()))

            error_list = [error_at_t.sum() for error_at_t in error_list]   # 是一个Variable的列表
            total_error = error_list[0] * time_loss_weights[0]
            for err, time_weight in zip(error_list[1:], time_loss_weights[1:]):
                total_error = total_error + err * time_weight

            loss = total_error
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if (step + 1) == 2500:
            #     zcr_state_dict = {
            #         'epoch'     : (e + 1),
            #         'tr_loss'   : 0,
            #         'state_dict': prednet.state_dict(),
            #         'optimizer' : optimizer.state_dict()
            #     }
            #     saveCheckpoint(zcr_state_dict)

            # print('epoch: [%3d/%3d] | step: [%4d/%4d]  loss: %.4f' % ((e + 1), args.epochs, (step + 1), len(dataLoader), loss.data[0]))

            tr_loss += loss.data[0]
            sum_trainLoss_in_epoch += loss.data[0]
            if step % printCircle == (printCircle - 1):
                print('epoch: [%3d/%3d] | [%4d/%4d]  loss: %.4f  lr: %.5lf' % ((e + 1), args.epochs, (step + 1), len(dataLoader), tr_loss / printCircle, optimizer.param_groups[0]['lr']))
                tr_loss = 0.0

        endTime_epoch = time.time()
        print('Time Consumed within an epoch: %.2f (s)' % (endTime_epoch - startTime_epoch))

        if sum_trainLoss_in_epoch < min_trainLoss_in_epoch:
            min_trainLoss_in_epoch = sum_trainLoss_in_epoch
            zcr_state_dict = {
                'epoch'     : (e + 1),
                'tr_loss'   : min_trainLoss_in_epoch,
                'state_dict': prednet.state_dict(),
                'optimizer' : optimizer.state_dict()
            }
            saveCheckpoint(zcr_state_dict)


def saveCheckpoint(zcr_state_dict, fileName = './checkpoint/checkpoint_newest.pkl'):
    '''save the checkpoint for both restarting and evaluating.'''
    tr_loss  = '%.4f' % zcr_state_dict['tr_loss']
    # val_loss = '%.4f' % zcr_state_dict['val_loss']
    epoch = zcr_state_dict['epoch']
    # fileName = './checkpoint/checkpoint_epoch' + str(epoch) + '_trLoss' + tr_loss + '_valLoss' + val_loss + '.pkl'
    fileName = '/media/sdb1/chenrui/checkpoint/PredNet/checkpoint_epoch' + str(epoch) + '_trLoss' + tr_loss + '.pkl'
    torch.save(zcr_state_dict, fileName)



if __name__ == '__main__':
    args = arg_parse()
    print_args(args)

    # DATA_DIR = args.dataPath
    # data_file = os.path.join(DATA_DIR, 'X_test.h5')
    # source_file = os.path.join(DATA_DIR, 'sources_test.h5')
    # output_mode = 'error'
    # sequence_start_mode = 'all'
    # N_seq = None
    # dataLoader = ZcrDataLoader(data_file, source_file, output_mode, sequence_start_mode, N_seq, args).dataLoader()

    # images, target = next(iter(dataLoader))
    # print(images)
    # print(target)

    n_channels = args.n_channels
    img_height = args.img_height
    img_width  = args.img_width

    # stack_sizes       = eval(args.stack_sizes)
    # R_stack_sizes     = eval(args.R_stack_sizes)
    # A_filter_sizes    = eval(args.A_filter_sizes)
    # Ahat_filter_sizes = eval(args.Ahat_filter_sizes)
    # R_filter_sizes    = eval(args.R_filter_sizes)

    stack_sizes       = (n_channels, 48, 96, 192)
    R_stack_sizes     = stack_sizes
    A_filter_sizes    = (3, 3, 3)
    Ahat_filter_sizes = (3, 3, 3, 3)
    R_filter_sizes    = (3, 3, 3, 3)

    prednet = PredNet(stack_sizes, R_stack_sizes, A_filter_sizes, Ahat_filter_sizes, R_filter_sizes,
                      output_mode = 'error', data_format = args.data_format, return_sequences = True)
    print(prednet)
    prednet.cuda()

    assert args.mode == 'train'
    train(prednet, args)

