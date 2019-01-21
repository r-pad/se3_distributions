python ../../src/generic_pose/training/finetune_ycb_trainer.py \
    --log_dir '/home/bokorn/results/ycb_finetune/01_002_master_chef_can/augmentation_rot' \
    --checkpoint_dir '/scratch/bokorn/results/ycb_finetune/002_master_chef_can/augmentation_rot' \
    --benchmark_folder '/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset' \
    --target_object 1 \
    --weight_file '/home/bokorn/pretrained/distance/shapenet_exp_fo20_th25.pth' \
    --augmentation_probability 0.5 \
    --brightness_jitter 1.0 \
    --contrast_jitter 1.0 \
    --saturation_jitter 1.0 \
    --hue_jitter 0.25 \
    --max_translation 0.2 \
    --min_scale 0.8 \
    --max_scale 1.2 \
    --max_num_occlusions 3 \
    --min_occlusion_area 0.1 \
    --max_occlusion_area 0.3 \
    --rotate_image \
    --top_n 1 \
    --batch_size 8 \
    --falloff_angle 20.0 --loss_type 'exp'

