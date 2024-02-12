#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

    # --train_list list/${DATASET}_list.txt \

# settings
MODEL_ARC=$1
NAME=$2
DATASET=$3
MEDMNIST_TASK_ID=$4
INITCLASS=$5
OUTPUT=results/${NAME}-${MODEL_ARC}/
mkdir -p ${OUTPUT}

# CUDA_LAUNCH_BLOCKING=1
python3 -u train_classify.py \
    --arch $MODEL_ARC \
    --name $NAME \
    --dataset $DATASET \
    --medmnist_task_id $MEDMNIST_TASK_ID \
    --initclass $INITCLASS \
    --increment 10 \
    --workers 8 \
    --epochs 30 \
    --start-epoch 0 \
    --batch-size 32 \
    --learning-rate 10e-5 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --print-freq 100 \
    --pth-save-fold ${OUTPUT} \
    --pth-save-epoch 30 \
    --parallel 0 \
    --dist-url 'tcp://localhost:10001' 2>&1 | tee ${OUTPUT}/output.log 


# bash run/train_classify.sh resnet18 cifar_classify cifar100 0 0