python ../../src/generic_pose/training/finetune_ycb_trainer.py \
    --results_dir '/scratch/bokorn/results/ycb_finetune/011_banana/shapenet_exp_fo20_th25' \
    --benchmark_folder '/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset' \
    --target_object 10 \
    --weight_file '/scratch/bokorn/results/shapenet/distance/shapenet_exp_fo20_th25/2018-08-03_02-29-12/weights/checkpoint_86000.pth' \
    --batch_size 16 --uniform_prop 1.0 \
    --falloff_angle 20.0 --loss_type 'exp'

