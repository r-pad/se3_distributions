python ../../src/generic_pose/training/feature_comparison_debug.py \
    --object_index 1 \
    --log_dir '/scratch/bokorn/results/debug/dense_fusion_feature_comparison_dropout_lr_'${1} \
    --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
    --feature_folder '/scratch/bokorn/results/dense_fusion_global_feat' \
    --feature_size 1024 \
    --num_epochs 100000 \
    --batch_size 2 \
    --falloff_angle 20.0 \
    --dropout \
    --lr ${1} 

