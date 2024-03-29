set -e

lr=1e-5

for i in {1..21}
do
    python ../src/se3_distributions/training/loglik_training.py \
        --object_id ${i} \
        --log_dir '/scratch/bokorn/results/log_lik/df_local_reg_orig/'${i}'/lr_'$lr \
        --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
        --feature_folder '/scratch/bokorn/results/dense_fusion_local_orig_feat/' \
        --feature_key 'feat' \
        --feature_size 1408 \
        --num_epochs 200 \
        --batch_size 24 \
        --interp_k 4\
        --num_augs 0 \
        --fill_with_exact \
        --lr $lr 
done
