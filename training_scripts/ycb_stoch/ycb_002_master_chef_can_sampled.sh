#!/bin/bash
SRC_DIR=$HOME/src/generic_pose/generic_pose/src/generic_pose/
echo ${1}
python $SRC_DIR/training/finetune_ycb_trainer.py \
    --log_dir '/home/bokorn/results/ycb_finetune/01_002_master_chef_can/'${1//[^A-Za-z0-9_-]/_} \
    --checkpoint_dir '/scratch/bokorn/results/ycb_finetune/002_master_chef_can/'${1//[^A-Za-z0-9_-]/_} \
    --benchmark_folder '/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset' \
    --target_object 1 \
    --weight_file '/home/bokorn/pretrained/distance/shapenet_exp_fo20_th25.pth' \
    --sampling_distribution 'lambda x: '${1} \
    --per_instance_sampling \
    --batch_size 16 --top_n 0 \
    --falloff_angle 20.0 --loss_type 'exp'

