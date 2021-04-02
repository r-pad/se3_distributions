set -e

lr=1e-5
topx=2000

for i in {1..21}
do
    python ../../../src/generic_pose/training/feature_comparison_training_aug.py \
        --object_index $i \
        --log_dir '/scratch/bokorn/results/all_ycb/df_local_global/'$i'/lr_'$lr'_top_x'$topx \
        --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
        --feature_folder '/scratch/bokorn/results/dense_fusion_local_feat' \
        --feature_key 'feat_global' \
        --feature_size 1024 \
        --num_epochs 1000000 \
        --batch_size 24 \
        --falloff_angle 20.0 \
        --num_augs 0 \
        --fill_with_exact \
        --dropout \
        --lr $lr \
        --weight_top $topx \
        --max_samples 100000
done
