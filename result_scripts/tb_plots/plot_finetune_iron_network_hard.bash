python ../../src/generic_pose/utils/plot_tensorboard.py \
    --image_prefix '/home/bokorn/results/linemod_finetune/iron/hard_b16_' \
    --log_dirs \
    '/home/bokorn/results/linemod_finetune/iron/linemod_iron_exp_b_16_uni_0_0/2018-10-09_13-34-53/logs/train/' \
    '/home/bokorn/results/linemod_finetune/iron/linemod_iron_exp_b_16_uni_0_0/2018-10-09_13-34-53/logs/holdout/' \
    '/home/bokorn/results/linemod_finetune/iron/linemod_iron_exp_b_16_uni_0_5/2018-10-09_13-38-20/logs/train/' \
    '/home/bokorn/results/linemod_finetune/iron/linemod_iron_exp_b_16_uni_0_5/2018-10-09_13-38-20/logs/holdout/' \
    '/home/bokorn/results/linemod_finetune/iron/linemod_iron_exp_b_16_uni_1_0/2018-10-09_19-33-16/logs/train/' \
    '/home/bokorn/results/linemod_finetune/iron/linemod_iron_exp_b_16_uni_1_0/2018-10-09_19-33-16/logs/holdout/' \
    --log_labels \
    'Hard (T)' \
    'Hard (V)' \
    'Half (T)' \
    'Half (V)' \
    'Uni  (T)' \
    'Uni  (V)' \
    --scalar_tags 'loss' 'dist_top' 'rank_gt' 'rank_top' \
    --ylabels 'Loss' 'Top Vertex Error (degrees)' 'Groundtruth Vertex Rank' 'Top Vertex Rank' \
    --titles 'Percent Hard Examples (Batchsize 16)' \
    --smoothing 0.6 --max_step 2000
