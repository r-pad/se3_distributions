set -e

lr=1e-5

python ../../src/generic_pose/training/bingham_training.py \
    --log_dir '/scratch/bokorn/results/log_lik/df_global_duel/lr_'$lr \
    --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
    --feature_folder '/scratch/datasets/ycb/' \
    --feature_key 'feat_global' \
    --feature_size 1024 \
    --duel_bingham \
    --num_epochs 1000000 \
    --batch_size 32 \
    --num_augs 0 \
    --fill_with_exact \
    --lr $lr \
