#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

    # --train_list list/${NAME}_list.txt \

# settings
MODEL_ARC=$1
NAME=$2
REPLAY=$3
MEAN=$4
INITCLASS=$5
PREV_WEIGHTS=$6
OUTPUT=results/${NAME}-${MODEL_ARC}/
mkdir -p ${OUTPUT}

# CUDA_LAUNCH_BLOCKING=1
python3 -u train.py \
    --arch $MODEL_ARC \
    --name $NAME \
    --replay $REPLAY \
    --mean $MEAN \
    --initclass $INITCLASS \
    --increment 10 \
    --prev_task_weights $PREV_WEIGHTS \
    --workers 32 \
    --epochs 100 \
    --start-epoch 0 \
    --batch-size 32 \
    --learning-rate 0.005 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --print-freq 1 \
    --pth-save-fold ${OUTPUT} \
    --pth-save-epoch 50 \
    --parallel 0 \
    --dist-url 'tcp://localhost:10001' 2>&1 | tee ${OUTPUT}/output.log 


# bash run/train.sh resnet18 name 1 0.5 0