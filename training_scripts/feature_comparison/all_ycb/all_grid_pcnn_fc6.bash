set -e

lr=1e-5
topx=2000

for i in {1..21}
do
    python ../../../src/generic_pose/training/feature_grid_training_trainval.py \
        --object_index ${i} \
        --log_dir '/scratch/bokorn/results/all_ycb/pcnn_fc6_sig_grid/'${i}'/trainval_no_augs' \
        --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
        --feature_folder '/scratch/bokorn/results/posecnn_feat_all' \
        --feature_key 'fc6' \
        --feature_size 4096 \
        --num_augs 0 \
        --num_epochs 1000000 \
        --batch_size 6 \
        --falloff_angle 20.0 \
        --lr 1e-6 
done
