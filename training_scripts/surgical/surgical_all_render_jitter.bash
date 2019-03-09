#!/bin/bash
SRC_DIR=$HOME/src/generic_pose/generic_pose/src/generic_pose/
echo ${1}
mkdir -p /scratch/bokorn/data/demo/surgical/${1}/train
mkdir -p /scratch/bokorn/data/demo/surgical/${1}/valid

python $SRC_DIR/training/render_trainer.py \
    --object_model '/home/bokorn/data/surgical_tools/models/'${1}'/'${1}'.obj' \
    --renders_folder '/scratch/bokorn/data/demo/surgical/'${1} \
    --log_dir '/home/bokorn/results/sugical/'${1}'_jitter' \
    --checkpoint_dir '/scratch/bokorn/results/sugical/'${1}'_jitter' \
    --random_render_offset \
    --weight_file '/home/bokorn/pretrained/distance/shapenet_exp_fo20_th25.pth' \
    --per_instance_sampling \
    --augmentation_probability 0.5 \
    --brightness_jitter 0.2 \
    --contrast_jitter 0.2 \
    --saturation_jitter 0.2 \
    --hue_jitter 0.05 \
    --rotate_image \
    --num_epochs 10000000 \
    --batch_size 16 --top_n 1 \
    --falloff_angle 20.0 --loss_type 'exp'

