python ../../src/generic_pose/training/feature_comparison_training.py \
    --object_index ${1} \
    --log_dir '/scratch/bokorn/results/dense_fusion_feature_comparison_results' \
    --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
    --feature_folder '/scratch/bokorn/results/dense_fusion_global_feat' \
    --feature_size 1024 \
    --num_epochs 1000000 \
    --batch_size 24 \
    --falloff_angle 20.0 \
    --lr 1e-4 

