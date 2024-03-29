set -e

lr=1e-5

python ../src/se3_distributions/training/bingham_training.py \
    --log_dir '/scratch/bokorn/results/log_lik/df_global_iso/lr_'$lr \
    --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
    --feature_folder '/scratch/datasets/ycb/' \
    --feature_key 'feat_global' \
    --feature_size 1024 \
    --num_epochs 1000000 \
    --batch_size 20 \
    --num_augs 0 \
    --fill_with_exact \
    --lr $lr \
