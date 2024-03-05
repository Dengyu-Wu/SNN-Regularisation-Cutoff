#!/bin/bash

python  ./scripts/training.py \
        base.batch_size=128 \
        base.epochs=100 \
        base.gpu_id=\'1\' \
        base.seed=3407 \
        base.port=\'11332\' \
        base.data=\'cifar10-dvs\' \
        base.model=\'vggann\' \
        base.dataset_path=\'/LOCAL/dengyu/dvs_dataset/dvs-cifar10\'\
        \
        snn-train.method=\'snn\' \
        snn-train.ann_layers=\'none\' \
        snn-train.snn_layers=\'baselayer\' \
        snn-train.regularizer=\'rcs\' \
        snn-train.loss='tet' \
        snn-train.multistep=True \
        snn-train.T=10 \
        snn-train.alpha=0.0