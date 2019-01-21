#!/bin/bash
SRC_DIR=$HOME/src/generic_pose/generic_pose/src/generic_pose/
echo ${1}
python $SRC_DIR/training/finetune_ycb_live_renders_trainer.py \
    --log_dir '/home/bokorn/results/ycb_finetune/01_002_master_chef_can/live_renders'${1} \
    --checkpoint_dir '/scratch/bokorn/results/ycb_finetune/002_master_chef_can/live_renders'${1} \
    --benchmark_folder '/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset' \
    --render_proportion ${1} \
    --target_object 1 \
    --weight_file '/home/bokorn/pretrained/distance/shapenet_exp_fo20_th25.pth' \
    --per_instance_sampling \
    --batch_size 16 --top_n 0  \
    --falloff_angle 20.0 --loss_type 'exp'

