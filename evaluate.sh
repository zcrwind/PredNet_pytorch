#!/bin/bash

# usage:
# 	./evaluate.sh

echo "Evaluate..."
mode='evaluate'

# @200.121
DATA_DIR='/media/sdb1/chenrui/kitti_data/h5/'
# Where results (prediction plots and evaluation file) will be saved.
RESULTS_SAVE_DIR='./kitti_results/'
checkpoint_file='./checkpoint/checkpoint_epoch1_trLoss1342.3278.pkl'	# load weights from checkpoint file for evaluating.

batch_size=10
num_plot=40		# how many images to plot.

# number of timesteps used for sequences in evaluating
num_timeSteps=10

workers=4
shuffle=false

data_format='channels_first'
n_channels=3
img_height=128
img_width=160

CUDA_VISIBLE_DEVICES=2 python evaluate.py \
	--mode ${mode} \
	--dataPath ${DATA_DIR} \
	--resultsPath ${RESULTS_SAVE_DIR} \
	--checkpoint_file ${checkpoint_file} \
	--batch_size ${batch_size} \
	--num_plot ${num_plot} \
	--num_timeSteps ${num_timeSteps} \
	--workers ${workers} \
	--shuffle ${shuffle} \
	--data_format ${data_format} \
	--n_channels ${n_channels} \
	--img_height ${img_height} \
	--img_width ${img_width}
	# --stack_sizes ${stack_sizes} \
	# --R_stack_sizes ${R_stack_sizes} \
	# --A_filter_sizes ${A_filter_sizes} \
	# --Ahat_filter_sizes ${Ahat_filter_sizes} \
	# --R_filter_sizes ${R_filter_sizes} \