set -e

lr=1e-5
topx=500

python ../src/se3_distributions/training/multiobject_feature_comparison_training_trainval.py \
    --log_dir '/scratch/bokorn/results/multi_object/df_local_global/trainval_lr_'$lr'_top_x'$topx \
    --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
    --feature_folder '/scratch/bokorn/results/dense_fusion_local_orig_feat' \
    --feature_key 'feat_global' \
    --feature_size 1024 \
    --num_epochs 1000000 \
    --batch_size 24 \
    --falloff_angle 20.0 \
    --num_augs 0 \
    --fill_with_exact \
    --dropout \
    --lr $lr \
    --weight_top $topx \
