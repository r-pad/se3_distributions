set -e

lr=1e-6

python ../../../src/generic_pose/training/bingham_training.py \
    --weight_file '/scratch/bokorn/results/log_lik/df_local_full_orig_duel/lr_1e-5/2019-09-08_16-16-03/weights/best_quat.pth' \
    --log_dir '/scratch/bokorn/results/log_lik/df_local_full_orig_duel/lr_'$lr \
    --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
    --feature_folder '/scratch/bokorn/results/dense_fusion_full_orig_feat' \
    --feature_key 'feat' \
    --feature_size 1408 \
    --duel_bingham \
    --num_epochs 1000000 \
    --batch_size 32 \
    --num_augs 0 \
    --fill_with_exact \
    --lr $lr \
