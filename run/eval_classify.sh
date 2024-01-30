#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# settings
MODEL_ARC=$1
CKPT=$2
INITCLASS=$3


# CUDA_LAUNCH_BLOCKING=1
python3 -u eval.py \
    --arch $MODEL_ARC \
    --initclass $INITCLASS \
    --increment 10 \
    --workers 8 \
    --batch-size 1 \
    --print-freq 10 \
    --resume ${CKPT} 


# bash run/eval.sh resnet18 results/cifar100_class-resnet18/100.pth 1 0.5 0