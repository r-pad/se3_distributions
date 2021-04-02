python ../../src/generic_pose/training/feature_comparison_training_aug.py \
    --object_index 1 \
    --log_dir '/scratch/bokorn/results/dense_fusion_lr_search/train_aug_offset_dropout_restart_'${1}'_'${2}'_top_x'${3} \
    --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
    --feature_folder '/scratch/bokorn/results/dense_fusion_global_feat' \
    --weight_file '/scratch/bokorn/results/dense_fusion_lr_search/train_aug_offset_dropout_adam_1e-5_top_x500/2019-06-13_23-40-26/002_master_chef_can/weights/best_quat.pth' \
    --feature_size 1024 \
    --num_epochs 1000000 \
    --batch_size 24 \
    --falloff_angle 20.0 \
    --num_augs 20 \
    --fill_with_exact \
    --dropout \
    --optimizer ${1} \
    --lr ${2} \
    --weight_top ${3} \


