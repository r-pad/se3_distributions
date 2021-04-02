set -e

lr=1e-5

python ../src/se3_distributions/training/loglik_training_linemod.py \
    --log_dir '/scratch/bokorn/results/log_lik_linemod/df_global_reg/lr_'$lr \
    --dataset_folder '/ssd0/datasets/linemod/Linemod_preprocessed/' \
    --feature_folder '/scratch/datasets/linemod/' \
    --feature_key 'feat_global' \
    --feature_size 1024 \
    --num_epochs 1000000 \
    --batch_size 24 \
    --fill_with_exact \
    --interp_k 4\
    --lr $lr \
