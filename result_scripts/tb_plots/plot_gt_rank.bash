echo ${1} 
echo ${2}
python ../../src/generic_pose/utils/plot_tensorboard.py \
    --image_prefix ${1} \
    --log_dirs \
    ${1}'/logs/train' \
    ${1}'/logs/valid' \
    '/home/bokorn/results/ycb_finetune/01_002_master_chef_can/top_0/2018-12-26_18-29-39/logs/train' \
    '/home/bokorn/results/ycb_finetune/01_002_master_chef_can/top_0/2018-12-26_18-29-39/logs/valid' \
    --log_labels \
    'Train' \
    'Valid' \
    'Baseline Train' \
    'Baseline Valid' \
    --scalar_tags 'rank_gt' \
    --ylabels 'Groundtruth Rank' \
    --max_step ${2}
