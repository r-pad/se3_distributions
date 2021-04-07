set -e

lr=1e-6

python ../src/se3_distributions/training/loglik_training.py \
    --weight_file '/scratch/bokorn/results/log_lik/posecnn_fc6_comp/lr_1e-6/2019-09-06_14-08-44/weights/checkpoint_9000.pth' \
    --log_dir '/scratch/bokorn/results/log_lik/posecnn_fc6_comp/lr_'$lr \
    --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
    --feature_folder '/scratch/bokorn/results/posecnn_feat_all' \
    --feature_key 'fc6' \
    --feature_size 4096 \
    --num_epochs 1000000 \
    --use_comparison \
    --batch_size 6 \
    --num_augs 0 \
    --fill_with_exact \
    --interp_k 4\
    --lr $lr \
