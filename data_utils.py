# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import datasets, transforms

import numpy as np
import h5py
import re


class SequenceGenerator(data.Dataset):
    """
    Sequence Generator

    the role of SequenceGenerator is equal to ImageFolder class in pytorch.

    the X_train.h5 contains 41396 images for 57 videos.
    the  X_test.h5 contains   832 images for  3 videos.
    the   X_val.h5 contains   154 images for  1 videos.

    Args:
        - data_file:
            data path, e.g., '/media/sdb1/chenrui/kitti_data/h5/X_train.h5'
        - source_file:
            e.g., '/media/sdb1/chenrui/kitti_data/h5/sources_train.h5'
            source for each image so when creating sequences can assure that consecutive frames are from same video.
                the content is like: 'road-2011_10_03_drive_0047_sync'
        - num_timeSteps:
            number of timesteps to predict
        - seed:
            Random seeding for data shuffling.
        - shuffle:
            shuffle or not
        - output_mode:
            `error` or `prediction`
        - sequence_start_mode:
            `all` or `unique`.
            `all`: allow for any possible sequence, starting from any frame.
            `unique`: create sequences where each unique frame is in at most one sequence
        - N_seq:
            TODO
    """
    def __init__(self, data_file, source_file, num_timeSteps, shuffle = False, seed = None,
                 output_mode = 'error', sequence_start_mode = 'all', N_seq = None, data_format = 'channels_first'):
        super(SequenceGenerator, self).__init__()
        pattern = re.compile(r'.*?h5/(.+?)\.h5')
        resList = re.findall(pattern, data_file)
        varName = resList[0]
        h5f = h5py.File(data_file, 'r')
        self.X = h5f[varName][:]    # X will be like (n_images, cols, rows, channels)

        resList = re.findall(pattern, source_file)
        varName = resList[0]
        source_h5f = h5py.File(source_file, 'r')
        self.sources = source_h5f[varName][:]   # list

        self.num_timeSteps = num_timeSteps
        self.shuffle = shuffle
        self.seed = seed
        assert output_mode in {'error', 'prediction'}
        self.output_mode = output_mode
        assert sequence_start_mode in {'all', 'unique'}
        self.sequence_start_mode = sequence_start_mode
        self.N_seq = N_seq
        self.data_format = data_format
        if self.data_format == 'channels_first':
            self.X = np.transpose(self.X, (0, 3, 1, 2))
        self.img_shape = self.X[0].shape
        self.num_samples = self.X.shape[0]

        if self.sequence_start_mode == 'all':       # allow for any possible sequence, starting from any frame (如果视频中任意一帧都可以作为起点,只需要确定加上序列长度后的小片段终点是否还属于同一个视频即可)
            self.possible_starts = np.array([i for i in range(self.num_samples - self.num_timeSteps) if self.sources[i] == self.sources[i + self.num_timeSteps - 1]])
        elif self.sequence_start_mode == 'unique':  # create sequences where each unique frame is in at most one sequence
            curr_location = 0
            possible_starts = []
            while curr_location < self.num_samples - self.num_timeSteps + 1:
                if self.sources[curr_location] == self.sources[curr_location + self.num_timeSteps - 1]:
                    possible_starts.append(curr_location)
                    curr_location += self.num_timeSteps
                else:
                    curr_location += 1
            self.possible_starts = possible_starts

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)

        if N_seq is not None and len(self.possible_starts) > N_seq:     # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:N_seq]
        self.N_sequences = len(self.possible_starts)                    # 所有可能的训练片段数

    def __getitem__(self, index):
        '''
        Args:
            index (int): Index

        Returns:
            tuple: (stacked images, target) where target is NOT class_index of the target class
                BUT the order of frames in sorting task.
        '''
        idx = self.possible_starts[index]
        image_group = self.preprocess(self.X[idx : (idx + self.num_timeSteps)])
        
        if self.output_mode == 'error':
            target = 0.             # model outputs errors, so y should be zeros
        elif self.output_mode == 'prediction':
            target = image_group    # output actual pixels

        return image_group, target

    def preprocess(self, X):
        return X.astype(np.float32) / 255.

    def __len__(self):
        return self.N_sequences

    def create_all(self):
        '''等价于原代码中的create_all. 为evaluate模式服务, 返回全部的测试数据.'''
        X_all = np.zeros((self.N_sequences, self.num_timeSteps) + self.img_shape, np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.X[idx : (idx + self.num_timeSteps)])
        return X_all


class ZcrDataLoader(object):
    '''[DataLoader for video frame predictation]'''
    def __init__(self, data_file, source_file, output_mode, sequence_start_mode, N_seq, args):
        super(ZcrDataLoader, self).__init__()
        self.data_file = data_file
        self.source_file = source_file
        self.output_mode = output_mode
        self.sequence_start_mode = sequence_start_mode
        self.N_seq = N_seq
        self.args = args

    def dataLoader(self):
        image_dataset = SequenceGenerator(self.data_file, self.source_file, self.args.num_timeSteps, self.args.shuffle, None, self.output_mode, self.sequence_start_mode, self.N_seq, self.args.data_format)
        # NOTE: 将drop_last设置为True, 可以删除最后一个不完整的batch(e.g.,当数据集大小不能被batch_size整除时, 最后一个batch的样本数是不够一个batch_size的, 这可能会导致某些要用到上一次结果的代码因为旧size和新size不匹配而报错(PredNet就有这个问题, 故这里将drop_last设置为True))
        dataloader = data.DataLoader(image_dataset, batch_size = self.args.batch_size, shuffle = False, num_workers = self.args.workers, drop_last = True)
        return dataloader


if __name__ == '__main__':
    pass