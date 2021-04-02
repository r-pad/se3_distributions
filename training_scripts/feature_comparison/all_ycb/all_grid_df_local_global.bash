set -e

lr=1e-5
topx=2000

for i in {1..21}
do
    python ../../../src/generic_pose/training/feature_grid_training.py \
        --object_index ${i} \
        --log_dir '/scratch/bokorn/results/all_ycb/df_local_global_grid/'${i}'/no_augs' \
        --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
        --feature_folder '/scratch/bokorn/results/dense_fusion_local_feat' \
        --feature_key 'feat_global' \
        --feature_size 1024 \
        --num_augs 0 \
        --num_epochs 1000000 \
        --batch_size 24 \
        --falloff_angle 20.0 \
        --lr 1e-5 
done
