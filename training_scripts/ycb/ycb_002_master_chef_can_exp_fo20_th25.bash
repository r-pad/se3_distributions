python ../../src/generic_pose/training/finetune_ycb_trainer.py \
    --results_dir '/scratch/bokorn/results/ycb_finetune/002_master_chef_can/shapenet_exp_fo20_th25' \
    --benchmark_folder '/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset' \
    --target_object 1 \
    --weight_file '/home/bokorn/results/ycb_finetune/002_master_chef_can/shapenet_exp_fo20_th25/2018-11-19_15-27-28/weights/best_quat.pth' \
    --batch_size 16 --uniform_prop 1.0 \
    --falloff_angle 20.0 --loss_type 'exp'

