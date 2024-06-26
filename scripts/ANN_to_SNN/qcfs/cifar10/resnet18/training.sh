#!/bin/bash

python  ./scripts/training.py \
        base.batch_size=128 \
        base.epochs=300 \
        base.gpu_id=\'0\' \
        base.seed=200 \
        base.port=\'20302\' \
        base.data=\'cifar10\' \
        base.model=\'resnet18\' \
        base.dataset_path=\'datasets\' \
        \
        snn-train.method=\'ann\' \
        snn-train.ann_layers=\'qcfs\' \
        snn-train.regularizer=\'none\' \
        snn-train.multistep_ann=False \
        snn-train.add_time_dim=True \
        snn-train.T=1 \
        snn-train.L=4 \
        snn-train.alpha=0.001