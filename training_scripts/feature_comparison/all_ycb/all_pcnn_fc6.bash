set -e 

lr=1e-6
topx=2000

for i in {10..21}
do
    python ../../../src/generic_pose/training/feature_comparison_training_aug.py \
        --object_index $i \
        --log_dir '/scratch/bokorn/results/all_ycb_train_valid/posecnn_fc6/'$i'/lr_'$lr'_top_x'$topx \
        --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
        --feature_folder '/scratch/bokorn/results/posecnn_feat_all' \
        --feature_key 'fc6' \
        --feature_size 4096 \
        --num_epochs 1000000 \
        --batch_size 6 \
        --falloff_angle 20.0 \
        --num_augs 0 \
        --fill_with_exact \
        --dropout \
        --lr $lr \
        --weight_top $topx \
        --max_samples 100000
done

