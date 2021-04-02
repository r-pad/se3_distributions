set -e

lr=1e-6

python ../../../src/generic_pose/training/bingham_training.py \
    --log_dir '/scratch/bokorn/results/log_lik/posecnn_fc6_duel/lr_'$lr \
    --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
    --feature_folder '/scratch/bokorn/results/posecnn_feat_all' \
    --feature_key 'fc6' \
    --feature_size 4096 \
    --duel_bingham \
    --num_epochs 1000000 \
    --batch_size 32 \
    --num_augs 0 \
    --fill_with_exact \
    --lr $lr \
