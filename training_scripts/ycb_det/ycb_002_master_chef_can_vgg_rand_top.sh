#!/bin/bash
SRC_DIR=$HOME/src/generic_pose/generic_pose/src/generic_pose/
echo ${1}
python $SRC_DIR/training/finetune_ycb_trainer.py \
    --log_dir '/home/bokorn/results/ycb_finetune/01_002_master_chef_can/vgg_rand_top_'${1} \
    --checkpoint_dir '/scratch/bokorn/results/ycb_finetune/002_master_chef_can/vgg_rand_top_'${1} \
    --benchmark_folder '/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset' \
    --target_object 1 \
    --per_instance_sampling \
    --model_type 'vgg16' \
    --random_init \
    --num_indices 24 \
    --image_chunk_size 100 \
    --num_workers 4 --batch_size 4 --top_n ${1}  \
    --falloff_angle 20.0 --loss_type 'exp'

