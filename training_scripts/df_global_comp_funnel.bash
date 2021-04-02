set -e

lr=1e-5

python ../src/se3_distributions/training/loglik_training_funnel.py \
    --log_dir '/scratch/bokorn/results/log_lik/df_global_comp_funnel/lr_'$lr \
    --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
    --feature_folder '/scratch/datasets/ycb/' \
    --feature_key 'feat_global' \
    --feature_size 1024 \
    --use_comparison \
    --num_epochs 100000 \
    --batch_size 24 \
    --num_augs 0 \
    --fill_with_exact \
    --interp_k 4\
    --lr $lr \
