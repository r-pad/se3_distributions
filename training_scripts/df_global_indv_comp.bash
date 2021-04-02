set -e

lr=1e-5

for i in {1..21}
do
    python ../src/generic_pose/training/loglik_training.py \
        --object_id ${i} \
        --log_dir '/scratch/bokorn/results/log_lik/df_global_orig_comp/'${i}'/lr_'$lr \
        --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
        --feature_folder '/scratch/bokorn/results/dense_fusion_local_orig_feat/' \
        --feature_key 'feat_global' \
        --feature_size 1024 \
        --use_comparison \
        --num_epochs 20 \
        --batch_size 24 \
        --num_augs 0 \
        --fill_with_exact \
        --interp_k 4\
        --lr $lr 
done
