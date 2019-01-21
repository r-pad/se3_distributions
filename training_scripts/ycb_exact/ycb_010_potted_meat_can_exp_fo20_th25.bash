python ../../src/generic_pose/training/finetune_ycb_trainer.py \
    --results_dir '/scratch/bokorn/results/ycb_finetune/010_potted_meat_can/shapenet_exp_fo20_th25' \
    --benchmark_folder '/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset' \
    --target_object 9 \
    --weight_file '/home/bokorn/results/ycb_finetune/010_potted_meat_can/weights/checkpoint_18000.pth' \
    --use_exact_render \
    --batch_size 16 --uniform_prop 1.0 \
    --falloff_angle 20.0 --loss_type 'exp'

