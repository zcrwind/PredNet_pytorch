# -*- coding: utf-8 -*-


'''
Usage:
    python visualization.py
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn.functional as F
from torch.autograd import Variable

def sortByVariance(filtersData):
    '''resort the filters by variance.'''
    sumedData = np.sum(filtersData, axis = 3)
    flat = sumedData.reshape(sumedData.shape[0], sumedData.shape[1] * sumedData.shape[2])
    std = np.std(flat, axis = 1)
    order = np.argsort(std)
    filterNum = int(order.shape[0] - (order.shape[0] % 10))     # e.g., 57——>50
    sortedData = np.zeros((filterNum,) + filtersData.shape[1:])
    for i in range(filterNum):
        sortedData[i, :, :, :] = filtersData[order[i], :, :, :]
    return sortedData

def visualize(filtersData, output_figName):
    '''
    visualize the conv1 filters
    filtersData: (filters_num, height, width, 3)
    '''
    print(output_figName)
    filtersData = np.squeeze(filtersData)
    print('after squeeze: ', filtersData.shape)     # (96, 11, 11, 3)

    # normalize filtersData for display
    filtersData = (filtersData - filtersData.min()) / (filtersData.max() - filtersData.min())
    filtersData = sortByVariance(filtersData)
    print('after sorting: ', filtersData.shape)     # (96, 11, 11, 3)

    filters_num = filtersData.shape[0]
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(filters_num)))
    # add some space between filters
    padding = (((0, 0), (0, 1), (0, 1)) + ((0, 0),) * (filtersData.ndim - 3))   # don't pad the last dimension (if there is one)
    # padding = (((0, 64 - filters_num), (0, 1), (0, 1)) + ((0, 0),) * (filtersData.ndim - 3))   # don't pad the last dimension (if there is one)
    print(padding)  # ((0, 0), (0, 1), (0, 1), (0, 0))
    filtersData = np.pad(filtersData, padding, mode = 'constant', constant_values = 1)  # pad with ones (white)
    print('after padding: ', filtersData.shape)     # (96, 12, 12, 3)
    # tile the filters into an image
    filtersData = filtersData.reshape((5, 10) + filtersData.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, filtersData.ndim + 1)))
    print('after reshape1: ', filtersData.shape)    # (6, 12, 16, 12, 3)
    # filtersData = filtersData.reshape((8, 8) + filtersData.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, filtersData.ndim + 1)))
    filtersData = filtersData.reshape((5 * filtersData.shape[1], 10 * filtersData.shape[3]) + filtersData.shape[4:])
    print('after reshape2: ', filtersData.shape)    # (72, 192, 3)
    # filtersData = filtersData.reshape((8 * filtersData.shape[1], 8 * filtersData.shape[3]) + filtersData.shape[4:])
    
    plt.imshow(filtersData)
    plt.axis('off')
    plt.savefig(output_figName, bbox_inches = 'tight')

def get_filtersData(checkpoint_file):
    '''get the filters data from checkpoint file.'''
    checkpoint = torch.load(checkpoint_file)
    stateDict = checkpoint['state_dict']
    ## debug
    # for k, v in stateDict.items():
    #     print(k)
    conv1_filters = stateDict['feature.0.weight']
    conv1_filters = conv1_filters.cpu().numpy() # if no `.cpu()`: RuntimeError: can't convert CUDA tensor to numpy (it doesn't support GPU arrays). Use .cpu() to move the tensor to host memory first.
    conv1_filters = conv1_filters.transpose(0, 2, 3, 1)
    # print(conv1_filters.shape)  # (96, 11, 11, 12)
    return conv1_filters

def visualize_layer2(filtersData, output_figName):
    '''A.2.weight'''
    filtersData = np.squeeze(filtersData)
    print('after squeeze: ', filtersData.shape)

    # normalize filtersData for display
    filtersData = (filtersData - filtersData.min()) / (filtersData.max() - filtersData.min())

    sumedData = np.sum(filtersData, axis = 3)
    flat = sumedData.reshape(sumedData.shape[0], sumedData.shape[1] * sumedData.shape[2])
    std = np.std(flat, axis = 1)
    order = np.argsort(std)
    # filterNum = int(order.shape[0] - (order.shape[0] % 10))
    sortedData = np.zeros(filtersData.shape)
    for i in range(filtersData.shape[0]):
        sortedData[i, :, :, :] = filtersData[order[i], :, :, :]
    filtersData = sortedData
    print('after sorting: ', filtersData.shape)

    filters_num = filtersData.shape[0]
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(filters_num)))
    # add some space between filters
    padding = (((0, 0), (0, 1), (0, 1)) + ((0, 0),) * (filtersData.ndim - 3))   # don't pad the last dimension (if there is one)
    # padding = (((0, 64 - filters_num), (0, 1), (0, 1)) + ((0, 0),) * (filtersData.ndim - 3))   # don't pad the last dimension (if there is one)
    print(padding)  # ((0, 0), (0, 1), (0, 1), (0, 0))
    filtersData = np.pad(filtersData, padding, mode = 'constant', constant_values = 1)  # pad with ones (white)
    print('after padding: ', filtersData.shape)     # (96, 12, 12, 3)
    # tile the filters into an image
    filtersData = filtersData.reshape((3, 16) + filtersData.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, filtersData.ndim + 1)))
    print('after reshape1: ', filtersData.shape)    # (6, 12, 16, 12, 3)
    # filtersData = filtersData.reshape((8, 8) + filtersData.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, filtersData.ndim + 1)))
    filtersData = filtersData.reshape((3 * filtersData.shape[1], 16 * filtersData.shape[3]) + filtersData.shape[4:])
    print('after reshape2: ', filtersData.shape)    # (72, 192, 3)
    # filtersData = filtersData.reshape((8 * filtersData.shape[1], 8 * filtersData.shape[3]) + filtersData.shape[4:])
    
    plt.imshow(filtersData)
    plt.axis('off')
    plt.savefig(output_figName, bbox_inches = 'tight')



if __name__ == '__main__':
    state_dict_file = './model_data_keras2/preTrained_weights_forPyTorch.pkl'
    stateDict = torch.load(state_dict_file)
    modules = ['A', 'Ahat', 'c', 'f', 'i', 'o']
    # for m in modules:
    #     # kernel = stateDict[m + '.0.weight'].cpu().numpy()
    #     kernel = stateDict[m + '.0.weight'].cpu()
    #     # print(kernel.shape)
    #     # A: (48, 6, 3, 3)
    #     # Ahat: (3, 3, 3, 3)
    #     # c、f、i、o: (3, 57, 3, 3)
    #     # kernel = F.upsample(input = Variable(kernel), scale_factor = 2, mode = 'nearest')
    #     # kernel = F.upsample(input = Variable(kernel), scale_factor = 4, mode = 'nearest')
    #     # kernel = F.upsample(input = Variable(kernel), scale_factor = 2, mode = 'bilinear')
    #     # kernel = F.upsample(input = Variable(kernel), scale_factor = 4, mode = 'bilinear')
    #     # kernel = F.upsample(input = Variable(kernel), scale_factor = 2, mode = 'linear')  # 不行, linear只接受3D输入
    #     print(kernel.data.size())
    #     kernel = kernel.data.numpy()
    #     kernel = np.transpose(kernel, (1, 2, 3, 0))
    #     if m in ['c', 'f', 'i', 'o']:
    #         visualize(kernel, './conv1_filters/' + m + '.png')

    # kernel = stateDict['A.2.weight'].cpu()  # (96, 96, 3, 3)
    kernel = stateDict['Ahat.2.weight'].cpu()  # (48, 48, 3, 3)
    kernel = F.upsample(input = Variable(kernel), scale_factor = 4, mode = 'bilinear')
    kernel = kernel.data.numpy()
    kernel = np.transpose(kernel, (1, 2, 3, 0))[..., :3]    # orz...原来有96个'RGB通道', 无法显示成图像, 人为截取前三维
    print('before calling visualization func: ', kernel.shape)
    visualize_layer2(kernel, './conv1_filters/Ahat.2.kernel.png')
