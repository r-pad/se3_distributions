set -e

lr=1e-5

python ../src/generic_pose/training/loglik_training.py \
    --log_dir '/scratch/bokorn/results/log_lik/df_local_comp/lr_'$lr \
    --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
    --feature_folder '/scratch/datasets/ycb/' \
    --feature_key 'feat' \
    --feature_size 1408 \
    --use_comparison \
    --num_epochs 1000000 \
    --batch_size 20 \
    --num_augs 0 \
    --fill_with_exact \
    --interp_k 4\
    --lr $lr \
