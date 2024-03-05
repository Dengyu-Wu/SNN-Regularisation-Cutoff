#!/bin/bash

python  ./scripts/evaluation.py \
        base.batch_size=128 \
        base.epochs=300 \
        base.gpu_id=\'1\' \
        base.seed=1200 \
        base.port=\'11152\' \
        base.data=\'cifar10\' \
        base.model=\'resnet18\' \
        base.dataset_path=\'datasets\' \
        \
        snn-train.method=\'snn\' \
        snn-train.ann_layers=\'none\' \
        snn-train.snn_layers=\'baselayer\' \
        snn-train.regularizer=\'rcs\' \
        snn-train.TET=False \
        snn-train.multistep=False \
        snn-train.add_time_dim=True \
        snn-train.T=4 \
        snn-train.alpha=0.0 \
        \
        snn-test.epsilon=0.01 \
        snn-test.reset_mode='hard' \
        snn-test.model_path=\'outputs/cifar10-resnet18-snn2T4L-TETFalse-TEBNFalse-none-rcs-alpha0.0-seed1200-epochs300/cifar10.pth\' \
