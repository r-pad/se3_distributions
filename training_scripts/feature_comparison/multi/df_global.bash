set -e

lr=1e-5
topx=2000

python ../../../src/generic_pose/training/multiobject_feature_comparison_training_aug.py \
    --log_dir '/scratch/bokorn/results/multi_object/df_global/lr_'$lr'_top_x'$topx \
    --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
    --feature_folder '/scratch/bokorn/results/dense_fusion_global_feat' \
    --feature_size 1024 \
    --num_epochs 1000000 \
    --batch_size 24 \
    --falloff_angle 20.0 \
    --num_augs 0 \
    --fill_with_exact \
    --dropout \
    --lr $lr \
    --weight_top $topx \