python ../../src/generic_pose/training/feature_comparison_training_aug.py \
    --object_index ${1} \
    --log_dir '/scratch/bokorn/results/posecnn/'${1}'/lr_'${2}'_top_x'${3} \
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
    --lr ${2} \
    --weight_top ${3} \


