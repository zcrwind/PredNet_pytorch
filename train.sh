#!/bin/bash

# usage:
# 	./train.sh

echo "Train..."
mode='train'

# @200.121
DATA_DIR='/media/sdb1/chenrui/kitti_data/h5/'
checkpoint_savePath='./checkpoint/'
checkpoint_file='./checkpoint/'	# checkpoint file name for restarting.

epochs=1
batch_size=8
optimizer='Adam'
learning_rate=0.001
momentum=0.9
beta1=0.9
beta2=0.99

workers=4

# it is vital for restarting
checkpoint_file='./checkpoint/'
printCircle=100

data_format='channels_first'
n_channels=3
img_height=128
img_width=160

# stack_sizes="($n_channels, 48, 96, 192)"
# R_stack_sizes=$stack_sizes
# A_filter_sizes="(3, 3, 3)"
# Ahat_filter_sizes="(3, 3, 3, 3)"
# R_filter_sizes="(3, 3, 3, 3)"

layer_loss_weightsMode='L_0'
# layer_loss='L_all'

# number of timesteps used for sequences in training
num_timeSteps=10

shuffle=true

CUDA_VISIBLE_DEVICES=0 python train.py \
	--mode ${mode} \
	--dataPath ${DATA_DIR} \
	--checkpoint_savePath ${checkpoint_savePath} \
	--epochs ${epochs} \
	--batch_size ${batch_size} \
	--optimizer ${optimizer} \
	--lr ${learning_rate} \
	--momentum ${momentum} \
	--beta1 ${beta1} \
	--beta2 ${beta2} \
	--workers ${workers} \
	--checkpoint_file ${checkpoint_file} \
	--printCircle ${printCircle} \
	--data_format ${data_format} \
	--n_channels ${n_channels} \
	--img_height ${img_height} \
	--img_width ${img_width} \
	--layer_loss_weightsMode ${layer_loss_weightsMode} \
	--num_timeSteps ${num_timeSteps} \
	--shuffle ${shuffle}
	# --stack_sizes ${stack_sizes} \
	# --R_stack_sizes ${R_stack_sizes} \
	# --A_filter_sizes ${A_filter_sizes} \
	# --Ahat_filter_sizes ${Ahat_filter_sizes} \
	# --R_filter_sizes ${R_filter_sizes} \
