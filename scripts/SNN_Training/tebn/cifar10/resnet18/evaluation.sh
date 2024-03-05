#!/bin/bash

python  ./scripts/evaluation.py \
        base.batch_size=128 \
        base.epochs=100 \
        base.gpu_id=\'1\' \
        base.seed=1200 \
        base.port=\'11152\' \
        base.data=\'cifar10-dvs\' \
        base.model=\'vggann\' \
        base.dataset_path=\'/LOCAL/dengyu/dvs_dataset/dvs-cifar10\'\
        \
        snn-train.method=\'snn\' \
        snn-train.ann_layers=\'none\' \
        snn-train.snn_layers=\'baselayer\' \
        snn-train.regularizer=\'rcs\' \
        snn-train.TEBN=True \
        snn-train.multistep=True \
        snn-train.add_time_dim=False \
        snn-train.T=10 \
        snn-train.alpha=0.0 \
        \
        snn-test.epsilon=0.01 \
        snn-test.reset_mode='hard' \
        snn-test.model_path=\'outputs/cifar10-resnet18-snn4T4L-TETFalse-TEBNTrue-none-rcs-alpha0.002-seed1200-epochs300/cifar10.pth\'
        