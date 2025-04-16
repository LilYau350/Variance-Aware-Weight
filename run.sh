torchrun  --nproc_per_node=2 main.py --train True --eval True --data_dir './CelebA_64x64' --dataset 'CelebA' \
          --patch_size 4 --in_chans 3 --image_size 64 --num_classes 1 --model 'DiT-S' --mean_type 'EPSILON' \
          --lr 1e-3 --betas 0.9 0.999 --dropout 0.0 --drop_label_prob 0.0 --total_steps 500000 --batch_size 128 \
          --beta_schedule 'cosine' --loss_type 'MSE' --weight_type 'lambda' --sampler_type 'uniform' \
          --warmup_steps 5000 --cosine_decay False --class_cond True --parallel True --amp True --mapping False \
          --sample_timesteps 20 --guidance_scale 1.0 --save_step 100000 --eval_step 50000 \
          --fid_cache './reference_batches/fid_stats_celeba_train.npz'
