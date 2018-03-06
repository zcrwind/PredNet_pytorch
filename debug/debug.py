# -*- coding: utf-8 -*-

import os
import numpy as np
import h5py

from prednet import PredNet

# dataDir = '../coxlab-prednet-cc76248/kitti_data/'
# trainSet_path = os.path.join(dataDir, 'X_train.hkl')
# train_sources = os.path.join(dataDir, 'sources_train.hkl')
# testSet_path = os.path.join(dataDir, 'X_test.hkl')
# test_sources = os.path.join(dataDir, 'sources_test.hkl')

# @200.121
# dataDir = '/media/sdb1/chenrui/kitti_data/h5/'
# trainSet_path = os.path.join(dataDir, 'X_train.h5')
# train_sources = os.path.join(dataDir, 'sources_train.h5')
# testSet_path  = os.path.join(dataDir, 'X_test.h5')
# test_sources  = os.path.join(dataDir, 'sources_test.h5')


# h5f = h5py.File(testSet_path,'r')
# testSet = h5f['X_test'][:]

# print(testSet)
# print(type(testSet))    # <class 'numpy.ndarray'>
# print(testSet.shape)    # (832, 128, 160, 3)

from data_utils import SequenceGenerator

data_file = '/media/sdb1/chenrui/kitti_data/h5/X_test.h5'
source_file = '/media/sdb1/chenrui/kitti_data/h5/sources_test.h5'
nt = 10

# sg = SequenceGenerator(data_file, source_file, nt)

# print(next(sg))

n_channels = 3
stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
prednet = PredNet(stack_sizes, R_stack_sizes, A_filt_sizes, Ahat_filt_sizes, R_filt_sizes, output_mode='error', data_format = 'channels_first', return_sequences=True)

input_shape = (8, 3, 128, 160)
prednet.build(input_shape)
print('\n'.join(['%s:%s' % item for item in prednet.__dict__.items()]))
print('+' * 30)
print(prednet.conv_layers['ahat'][1].strides)