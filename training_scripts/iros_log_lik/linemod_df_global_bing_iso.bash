set -e

lr=1e-5

python ../../src/generic_pose/training/bingham_training_linemod.py \
    --log_dir '/scratch/bokorn/results/log_lik_linemod/df_global_iso/lr_'$lr \
    --dataset_folder '/ssd0/datasets/linemod/Linemod_preprocessed/' \
    --feature_folder '/scratch/datasets/linemod/' \
    --feature_key 'feat_global' \
    --feature_size 1024 \
    --num_epochs 1000000 \
    --batch_size 20 \
    --fill_with_exact \
    --lr $lr \
