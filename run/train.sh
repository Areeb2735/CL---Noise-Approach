#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

    # --train_list list/${NAME}_list.txt \

# settings
MODEL_ARC=$1
NAME=$2
# REPLAY=$3
MEAN=$3
# INITCLASS=$5
# PREV_WEIGHTS=$6
OUTPUT=results/${NAME}-${MODEL_ARC}/
mkdir -p ${OUTPUT}

# --learning-rate 0.005
# --weight-decay 1e-4

# CUDA_LAUNCH_BLOCKING=1
python3 -u train.py \
    --arch $MODEL_ARC \
    --name $NAME \
    --mean $MEAN \
    --initclass 0 \
    --increment 10 \
    --workers 8 \
    --epochs 200 \
    --start-epoch 0 \
    --batch-size 32 \
    --learning-rate 0.1 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --print-freq 25 \
    --pth-save-fold ${OUTPUT} \
    --pth-save-epoch 50 \
    --parallel 0 \
    --dist-url 'tcp://localhost:10001' 2>&1 | tee ${OUTPUT}/output.log 


# bash run/train.sh resnet18 name 1 0.5 0

# bash run/train.sh resnet18 try 0.5