#!/bin/bash

python  ./examples/training.py \
        base.batch_size=128 \
        base.epochs=100 \
        base.gpu_id=\'0\' \
        base.seed=3407 \
        base.port=\'23152\' \
        base.data=\'cifar10-dvs\' \
        base.model=\'vggann\' \
        base.dataset_path=\'/LOCAL/dengyu/dvs_dataset/dvs-cifar10\'\
        \
        snn-train.method=\'snn\' \
        snn-train.ann_constrs=\'baseconstrs\' \
        snn-train.snn_layers=\'baselayer\' \
        snn-train.regularizer=\'rcs\' \
        snn-train.TET=False \
        snn-train.TEBN=True \
        snn-train.multistep=True \
        snn-train.add_time_dim=False \
        snn-train.T=10 \
        snn-train.alpha=0.00