python ../../src/generic_pose/training/finetune_ycb_trainer.py \
    --results_dir '/scratch/bokorn/results/ycb_finetune/007_tuna_fish_can/shapenet_exp_fo20_th25' \
    --benchmark_folder '/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset' \
    --target_object 6 \
    --weight_file '/home/bokorn/results/ycb_finetune/007_tuna_fish_can/shapenet_exp_fo20_th25/2018-11-19_16-03-13/weights/checkpoint_3000.pth' \
    --batch_size 16 --uniform_prop 1.0 \
    --falloff_angle 20.0 --loss_type 'exp'

