python ../../src/generic_pose/training/multiobject_feature_comparison_training_aug.py \
    --log_dir '/scratch/bokorn/results/dense_fusion_multi/train_aug_offset_dropout_'${1}'_top_x'${2} \
    --dataset_folder '/ssd0/datasets/ycb/YCB_Video_Dataset' \
    --feature_folder '/scratch/bokorn/results/dense_fusion_global_feat' \
    --feature_size 1024 \
    --num_epochs 1000000 \
    --batch_size 24 \
    --falloff_angle 20.0 \
    --num_augs 10 \
    --fill_with_exact \
    --dropout \
    --optimizer 'adam' \
    --lr ${1} \
    --weight_top ${2} \


