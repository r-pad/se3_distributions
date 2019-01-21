python ../../src/generic_pose/utils/plot_tensorboard.py \
    --image_prefix '/home/bokorn/results/ycb_finetune/035_power_drill/' \
    --log_dirs \
    '/home/bokorn/results/ycb_finetune/035_power_drill/logs/train' \
    '/home/bokorn/results/ycb_finetune/035_power_drill/logs/valid' \
    --log_labels \
    'Train' \
    'Valid' \
    --scalar_tags 'loss' 'dist_top' 'rank_gt' 'rank_top' \
    --ylabels 'Loss' 'Top Error (degrees)' 'Groundtruth Rank' 'Top Scored Rank'\
    --wall_time
