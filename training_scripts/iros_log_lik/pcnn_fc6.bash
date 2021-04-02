set -e 

lr=1e-6
topx=500

python ../../../src/generic_pose/training/multiobject_feature_comparison_training_trainval.py \
    --log_dir '/scratch/bokorn/results/multi_object/pcnn_fc6/trainval_lr_'$lr'_top_x'$topx \
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

