set -e

lr=1e-4

for i in {1..21}
do
    python ../src/se3_distributions/training/loglik_training.py \
        --object_id ${i} \
        --log_dir '/scratch/bokorn/results/log_lik/pcnn_fc6_comp/'${i}'/lr_'$lr \
        --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
        --feature_folder '/scratch/bokorn/results/posecnn_feat_all' \
        --feature_key 'fc6' \
        --feature_size 4096 \
        --use_comparison \
        --num_epochs 20 \
        --batch_size 6 \
        --num_augs 0 \
        --fill_with_exact \
        --interp_k 4\
        --lr $lr 
done
