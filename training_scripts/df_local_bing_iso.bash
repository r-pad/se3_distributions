set -e

lr=1e-5

python ../src/se3_distributions/training/bingham_training.py \
    --log_dir '/scratch/bokorn/results/log_lik/df_local_full_orig_iso/lr_'$lr \
    --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
    --feature_folder '/scratch/bokorn/results/dense_fusion_full_orig_feat' \
    --feature_key 'feat' \
    --feature_size 1408 \
    --num_epochs 1000000 \
    --batch_size 20 \
    --num_augs 0 \
    --fill_with_exact \
    --lr $lr \
