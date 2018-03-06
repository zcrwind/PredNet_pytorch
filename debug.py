# -*- coding: utf-8 -*-

import os
import numpy as np
import h5py

# dataDir = '../coxlab-prednet-cc76248/kitti_data/'
# trainSet_path = os.path.join(dataDir, 'X_train.hkl')
# train_sources = os.path.join(dataDir, 'sources_train.hkl')
# testSet_path = os.path.join(dataDir, 'X_test.hkl')
# test_sources = os.path.join(dataDir, 'sources_test.hkl')

# @200.121
dataDir = '/media/sdb1/chenrui/kitti_data/h5/'
trainSet_path = os.path.join(dataDir, 'X_train.h5')
train_sources = os.path.join(dataDir, 'sources_train.h5')
testSet_path  = os.path.join(dataDir, 'X_test.h5')
test_sources  = os.path.join(dataDir, 'sources_test.h5')


h5f = h5py.File(testSet_path,'r')
testSet = h5f['X_test'][:]

# print(testSet)
# print(type(testSet))    # <class 'numpy.ndarray'>
# print(testSet.shape)    # (832, 128, 160, 3)