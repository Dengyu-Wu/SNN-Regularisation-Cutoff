#!/bin/bash

python  ./scripts/training.py \
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
        snn-train.loss='tet' \
        snn-train.multistep=True \
        snn-train.add_time_dim=True \
        snn-train.T=2 \
        snn-train.alpha=0.00