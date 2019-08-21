# PredNet_pytorch

An implement of PredNet in pytorch. See the paper [Deep predictive coding networks for video prediction and unsupervised learning](https://arxiv.org/abs/1605.08104) in ICLR 2017 for more details.

The [offical code](https://github.com/coxlab/prednet) is implemented via Keras, and the project website can be found at [https://coxlab.github.io/prednet/](https://coxlab.github.io/prednet/).

## Dataset
The preprocessed KITTI data can be obtained using `downlaod_data.sh from` in [offical code](https://github.com/coxlab/prednet).

## How to run
### Train model
```
sh train.sh
```
### Evaluate model
```
sh evaluate.sh
```

## Some example results
![example](./kitti_results/prediction_plots/use_pretrained_weights/plot_5.png)
