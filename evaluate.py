# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
from torch.autograd import Variable

# zcr lib
from prednet import PredNet
from data_utils import ZcrDataLoader

def arg_parse():
    desc = "Video Frames Predicting Task via PredNet."
    parser = argparse.ArgumentParser(description = desc)

    parser.add_argument('--mode', default = 'train', type = str,
                        help = 'train or evaluate (default: train)')
    parser.add_argument('--dataPath', default = '', type = str, metavar = 'PATH',
                        help = 'path to video dataset (default: none)')
    parser.add_argument('--resultsPath', default = '', type = str, metavar = 'PATH',
                        help = 'saving path to results of PredNet (default: none)')
    parser.add_argument('--checkpoint_file', default = '', type = str,
                        help = 'checkpoint file for evaluating. (default: none)')
    parser.add_argument('--batch_size', default = 32, type = int, metavar = 'N',
                        help = 'The size of batch')
    parser.add_argument('--num_plot', default = 40, type = int, metavar = 'N',
                        help = 'how many images to plot')
    parser.add_argument('--num_timeSteps', default = 10, type = int, metavar = 'N',
                        help = 'number of timesteps used for sequences in training (default: 10)')
    parser.add_argument('--workers', default = 4, type = int, metavar = 'N',
                        help = 'number of data loading workers (default: 4)')
    parser.add_argument('--shuffle', default = True, type = bool,
                        help = 'shuffle or not')
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
    
    args = parser.parse_args()
    return args

def print_args(args):
    print('-' * 50)
    for arg, content in args.__dict__.items():
        print("{}: {}".format(arg, content))
    print('-' * 50)


def evaluate(model, args):
    '''Evaluate PredNet on KITTI sequences'''
    prednet = model     # Now prednet is the testing model (to output predictions)

    DATA_DIR = args.dataPath
    RESULTS_SAVE_DIR = args.resultsPath
    test_file = os.path.join(DATA_DIR, 'X_test.h5')
    test_sources = os.path.join(DATA_DIR, 'sources_test.h5')

    output_mode = 'prediction'
    sequence_start_mode = 'unique'
    N_seq = None
    dataLoader = ZcrDataLoader(test_file, test_sources, output_mode, sequence_start_mode, N_seq, args).dataLoader()
    X_test = dataLoader.dataset.create_all()
    # print('X_test.shape', X_test.shape)       # (83, 10, 3, 128, 160)
    X_test = X_test[:8, ...]                    # to overcome `cuda runtime error: out of memory`
    batch_size = X_test.shape[0]
    X_groundTruth = np.transpose(X_test, (1, 0, 2, 3, 4))      # (timesteps, batch_size, 3, 128, 160)
    X_groundTruth_list = []
    for t in range(X_groundTruth.shape[0]):
        X_groundTruth_list.append(np.squeeze(X_groundTruth[t, ...]))    # (batch_size, 3, 128, 160)

    X_test = Variable(torch.from_numpy(X_test).float().cuda())

    if prednet.data_format == 'channels_first':
        input_shape = (batch_size, args.num_timeSteps, n_channels, img_height, img_width)
    else:
        input_shape = (batch_size, args.num_timeSteps, img_height, img_width, n_channels)
    initial_states = prednet.get_initial_states(input_shape)
    predictions = prednet(X_test, initial_states)
    # print(predictions)
    # print(predictions[0].size())    # torch.Size([8, 3, 128, 160])

    X_predict_list = [pred.data.cpu().numpy() for pred in predictions]  # length of X_predict_list is timesteps. 每个元素shape是(batch_size, 3, H, W)

    # Compare MSE of PredNet predictions vs. using last frame. Write results to prediction_scores.txt
    # MSE_PredNet  = np.mean((real_X[:, 1:  ] - pred_X[:, 1:])**2)    # look at all timesteps except the first
    # MSE_previous = np.mean((real_X[:,  :-1] - real_X[:, 1:])**2)
    # if not os.path.exists(RESULTS_SAVE_DIR):
    #     os.mkdir(RESULTS_SAVE_DIR)
    # score_file = os.path.join(RESULTS_SAVE_DIR, 'prediction_scores.txt')
    # with open(score_file, 'w') as f:
    #     f.write("PredNet MSE: %f\n" % MSE_PredNet)
    #     f.write("Previous Frame MSE: %f" % MSE_previous)

    # Plot some predictions
    if prednet.data_format == 'channels_first':
        X_groundTruth_list = [np.transpose(batch_img, (0, 2, 3, 1)) for batch_img in X_groundTruth_list]
        X_predict_list     = [np.transpose(batch_img, (0, 2, 3, 1)) for batch_img in X_predict_list]
    assert len(X_groundTruth_list) == len(X_predict_list) == args.num_timeSteps
    timesteps = args.num_timeSteps
    total_num = X_groundTruth_list[0].shape[0]
    height = X_predict_list[0].shape[1]
    width  = X_predict_list[0].shape[2]

    n_plot = args.num_plot
    if n_plot > total_num:
        n_plot = total_num
    aspect_ratio = float(height) / width
    plt.figure(figsize = (timesteps, (2 * aspect_ratio)))
    gs = gridspec.GridSpec(2, timesteps)
    gs.update(wspace = 0., hspace = 0.)
    plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/')
    if not os.path.exists(plot_save_dir):
        os.mkdir(plot_save_dir)
    plot_idx = np.random.permutation(total_num)[:n_plot]
    for i in plot_idx:
        for t in range(timesteps):
            ## plot the ground truth.
            plt.subplot(gs[t])
            plt.imshow(X_groundTruth_list[t][i, ...], interpolation = 'none')
            plt.tick_params(axis = 'both', which = 'both', bottom = 'off', top = 'off', left = 'off', right = 'off', labelbottom = 'off', labelleft = 'off')
            if t == 0:
                plt.ylabel('Actual', fontsize = 10)

            ## plot the predictions.
            plt.subplot(gs[t + timesteps])
            plt.imshow(X_predict_list[t][i, ...], interpolation = 'none')
            plt.tick_params(axis = 'both', which = 'both', bottom = 'off', top = 'off', left = 'off', right = 'off', labelbottom = 'off', labelleft = 'off')
            if t == 0:
                plt.ylabel('Predicted', fontsize = 10)

        plt.savefig(plot_save_dir +  'plot_' + str(i) + '.png')
        plt.clf()
    print('The plots are saved in "%s"! Have a nice day!' % plot_save_dir)


def checkpoint_loader(checkpoint_file):
    '''load the checkpoint for weights of PredNet.'''
    print('Loading...', end = '')
    checkpoint = torch.load(checkpoint_file)
    print('Done.')
    return checkpoint

def load_pretrained_weights(model, state_dict_file):
    '''直接使用从原作者提供的Keras版本的预训练好的PredNet模型中拿过来的参数'''
    model = model.load_state_dict(torch.load(state_dict_file))
    print('weights loaded!')
    return model

if __name__ == '__main__':
    args = arg_parse()
    print_args(args)

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
                      output_mode = 'prediction', data_format = args.data_format, return_sequences = True)
    print(prednet)
    prednet.cuda()

    # print('\n'.join(['%s:%s' % item for item in prednet.__dict__.items()]))
    # print(type(prednet.state_dict()))   # <class 'collections.OrderedDict'>
    # for k, v in prednet.state_dict().items():
    #     print(k, v.size())

    ## 使用自己训练的参数
    checkpoint_file = args.checkpoint_file
    try:
        checkpoint = checkpoint_loader(checkpoint_file)
    except Exception:
        raise(RuntimeError('Cannot load the checkpoint file named %s!' % checkpoint_file))
    state_dict = checkpoint['state_dict']
    prednet.load_state_dict(state_dict)

    ## 直接使用作者提供的预训练参数
    # state_dict_file = './model_data_keras2/preTrained_weights_forPyTorch.pkl'
    # # prednet = load_pretrained_weights(prednet, state_dict_file)   # 这种不work... why?
    # prednet.load_state_dict(torch.load(state_dict_file))

    assert args.mode == 'evaluate'
    evaluate(prednet, args)
