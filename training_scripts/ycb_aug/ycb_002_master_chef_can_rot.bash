python ../../src/generic_pose/training/finetune_ycb_trainer.py \
    --log_dir '/home/bokorn/results/ycb_finetune/01_002_master_chef_can/augmentation_rot_only' \
    --checkpoint_dir '/scratch/bokorn/results/ycb_finetune/002_master_chef_can/augmentation_rot_only' \
    --benchmark_folder '/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset' \
    --target_object 1 \
    --weight_file '/home/bokorn/pretrained/distance/shapenet_exp_fo20_th25.pth' \
    --augmentation_probability 0.5 \
    --rotate_image \
    --top_n 1 \
    --batch_size 8 \
    --falloff_angle 20.0 --loss_type 'exp'

