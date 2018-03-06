# -*- coding: utf-8 -*-

'''将以hdf5形式保存的原Keras版本的PredNet模型的参数加载到zcr复现的pytorch版本的模型中.'''

import os
import numpy as np
import h5py

import torch
# from torch.autograd import Variable


weights_file = './model_data_keras2/prednet_kitti_weights.hdf5'
weights_f = h5py.File(weights_file, 'r')

pred_weights = weights_f['model_weights']['pred_net_1']['pred_net_1']	# contains 23 item: 4x4(i,f,c,o for 4 layers) + 4(Ahat for 4 layers) + 3(A for 4 layers)

keras_items = ['bias:0', 'kernel:0']
pytorch_items = ['weight', 'bias']

keras_modules = ['a', 'ahat', 'c', 'f', 'i', 'o']
keras_modules = ['layer_' + m + '_' + str(i) for m in keras_modules for i in range(4)]
keras_modules.remove('layer_a_3')
assert len(keras_modules) == 4 * 4 + 4 + 3

pytorch_modules_1 = ['A', 'Ahat']
pytorch_modules_2 = ['c', 'f', 'i', 'o']
pytorch_modules_1 = [m + '.' + str(2 * i) + '.' + item for m in pytorch_modules_1 for i in range(4) for item in pytorch_items]
pytorch_modules_1.remove('A.6.weight')
pytorch_modules_1.remove('A.6.bias')
pytorch_modules_2 = [m + '.' + str(i) + '.' + item for m in pytorch_modules_2 for i in range(4) for item in pytorch_items]
pytorch_modules = pytorch_modules_1 + pytorch_modules_2
assert len(pytorch_modules) == (4 * 4 + 4 + 3) * 2

weight_dict = dict()

# 从h5文件加载过来的是<class 'numpy.ndarray'>类型的权重, 需要将其转换为cuda.Tensor
for i in range(len(keras_modules)):
	weight_dict[pytorch_modules[i * 2 + 1]] = pred_weights[keras_modules[i]]['bias:0'][:]
	# weight_dict[pytorch_modules[i * 2 + 1]] = pred_weights[keras_modules[i]]['bias:0']
	weight_dict[pytorch_modules[i * 2]] = np.transpose(pred_weights[keras_modules[i]]['kernel:0'][:], (3, 2, 1, 0))
	# weight_dict[pytorch_modules[i * 2]] = pred_weights[keras_modules[i]]['kernel:0']

for k, v in weight_dict.items():
	# print(k, v)
	# weight_dict[k] = Variable(torch.from_numpy(v).float().cuda())
	weight_dict[k] = torch.from_numpy(v).float().cuda()

fileName = './model_data_keras2/preTrained_weights_forPyTorch.pkl'
weights_gift_from_keras = torch.save(weight_dict, fileName)