#!/bin/bash
# This script is used for training, inference and benchmarking
# the iid baseline method on Stanford 2D3DS. Users could
# also modify from this script for their use case.
#
# Usage:
#   # From Adversarial_Structure_Matching/ directory.
#   bash bashscripts/stanford_scenes/normal/train.sh
#
#

# Set up parameters for training.
BATCH_SIZE=8
TRAIN_INPUT_SIZE=512,512
WEIGHT_DECAY=5e-4
LEARNING_RATE=1e-2
ITER_SIZE=1
NUM_STEPS=120000
NUM_CLASSES=13
DEPTH_UNIT=512
FOLD=fold_1

# Set up parameters for inference.
INFERENCE_INPUT_SIZE=540,540
INFERENCE_STRIDES=320,320
INFERENCE_SPLIT=val

# Set up path for saving models.
SNAPSHOT_DIR=snapshots/stanford_scenes/normal/unet_iid/p512_bs8_lr${LEARNING_RATE}/${FOLD}

# Set up the procedure pipeline.
IS_TRAIN=1
IS_INFERENCE=1
IS_BENCHMARK=1

# Update PYTHONPATH.
export PYTHONPATH=`pwd`:$PYTHONPATH

# Set up the data directory.
DATAROOT=/path/to/data

# Train.
if [ ${IS_TRAIN} -eq 1 ]; then
  python3 pyscripts/train/train_stanford_scenes.py\
    --snapshot_dir ${SNAPSHOT_DIR}\
    --data_list dataset/stanford_scenes/${FOLD}/train_id.txt\
    --data_dir ${DATAROOT}/\
    --batch_size ${BATCH_SIZE}\
    --save_pred_every 30000\
    --update_tb_every 200\
    --input_size ${TRAIN_INPUT_SIZE}\
    --learning_rate ${LEARNING_RATE}\
    --weight_decay ${WEIGHT_DECAY}\
    --iter_size ${ITER_SIZE}\
    --num_classes ${NUM_CLASSES}\
    --num_steps $(($NUM_STEPS+1))\
    --random_mirror\
    --random_crop\
    --not_restore_classifier\
    --is_training\
    --train_normal
fi

# Inference.
if [ ${IS_INFERENCE} -eq 1 ]; then
  python3 pyscripts/inference/inference_stanford_scenes.py\
    --data_dir ${DATAROOT}/\
    --data_list dataset/stanford_scenes/${FOLD}/${INFERENCE_SPLIT}_id.txt\
    --input_size ${INFERENCE_INPUT_SIZE}\
    --strides ${INFERENCE_STRIDES}\
    --restore_from ${SNAPSHOT_DIR}/model.ckpt-${NUM_STEPS}\
    --colormap misc/colormap2d3ds.mat\
    --num_classes ${NUM_CLASSES}\
    --ignore_label 255\
    --save_dir ${SNAPSHOT_DIR}/results/${INFERENCE_SPLIT}\
    --train_normal
fi

# Benchmark.
if [ ${IS_BENCHMARK} -eq 1 ]; then
  python3 pyscripts/benchmark/benchmark_stanford_scenes.py\
    --pred_dir ${SNAPSHOT_DIR}/results/${INFERENCE_SPLIT}\
    --gt_dir ${DATAROOT}/all_area/data/\
    --num_classes ${NUM_CLASSES}\
    --train_normal
fi
