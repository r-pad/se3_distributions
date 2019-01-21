#!/bin/bash -login

python /home/bokorn/src/generic_pose/generic_pose/src/generic_pose/training/finetune_ycb_trainer.py \
    --log_dir '/home/bokorn/results/ycb_finetune/01_002_master_chef_can/exact_real' \
    --checkpoint_dir '/scratch/bokorn/results/ycb_finetune/002_master_chef_can/exact_real' \
    --benchmark_folder '/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset' \
    --target_object 1 \
    --weight_file '/home/bokorn/pretrained/distance/shapenet_exp_fo20_th25.pth' \
    --use_exact_render \
    --batch_size 16 \
    --falloff_angle 20.0 --loss_type 'exp'

