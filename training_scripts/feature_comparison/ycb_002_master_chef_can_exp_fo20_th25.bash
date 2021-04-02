python ../../src/generic_pose/training/finetune_ycb_w_renders_trainer.py \
    --results_dir '/scratch/bokorn/results/ycb_finetune/002_master_chef_can/base_and_exact_renders' \
    --benchmark_folder '/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset' \
    --renders_folder '/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset/base_renders/002_master_chef_can/2/' \
    --target_object 1 \
    --weight_file '/home/bokorn/pretrained/distance/shapenet_exp_fo20_th25.pth' \
    --use_exact_render \
    --batch_size 16 --uniform_prop 1.0 \
    --falloff_angle 20.0 --loss_type 'exp'

