#!/bin/bash

################################################################################################################################
# Ablation study in CIFAR-10
################################################################################################################################

# linear schedule + mse loss
torchrun  --nproc_per_node=2 main.py --train True --eval True --data_dir '' --dataset 'CIFAR-10' \
          --in_chans 3 --image_size 32 --num_classes 10 --model 'ADM-32' \
          --lr 1e-4 --betas 0.9 0.999 --dropout 0.0 --drop_label_prob 0.0 --total_steps 200000 --batch_size 128 \
          --beta_schedule 'linear' --loss_type 'MSE' --weight_type'constant' \
          --warmup_steps 5000 --cosine_decay True --class_cond True --parallel True --use_fp16 False \
          --sample_timesteps 10 --guidance_scale 1.5 --eps_scaler 1.0 --save_step 50000 --eval_step 50000 --fid_cache '' \

# consine schedule + mse loss
torchrun  --nproc_per_node=2 main.py --train True --eval True --data_dir '' --dataset 'CIFAR-10' \
          --in_chans 3 --image_size 32 --num_classes 10 --model 'ADM-32' \
          --lr 1e-4 --betas 0.9 0.999 --dropout 0.0 --drop_label_prob 0.0 --total_steps 200000 --batch_size 128 \
          ---beta_schedule 'cosine' --loss_type 'MSE' --weight_type'constant' \
          --warmup_steps 5000 --cosine_decay True --class_cond True --parallel True --use_fp16 False \
          --sample_timesteps 10 --guidance_scale 1.5 --eps_scaler 1.0 --save_step 50000 --eval_step 50000 --fid_cache '' \

# optim schedule + mse loss
torchrun  --nproc_per_node=2 main.py --train True --eval True --data_dir '' --dataset 'CIFAR-10' \
          --in_chans 3 --image_size 32 --num_classes 10 --model 'ADM-32' \
          --lr 1e-4 --betas 0.9 0.999 --dropout 0.0 --drop_label_prob 0.0 --total_steps 200000 --batch_size 128 \
          --beta_schedule 'optim_2' --loss_type 'MSE' --weight_type'constant' \
          --warmup_steps 5000 --cosine_decay True --class_cond True --parallel True --use_fp16 False \
          --sample_timesteps 10 --guidance_scale 1.5 --eps_scaler 1.0 --save_step 50000 --eval_step 50000 --fid_cache '' \

# linear schedule + mapped mse loss
torchrun  --nproc_per_node=2 main.py --train True --eval True --data_dir '' --dataset 'CIFAR-10' \
          --in_chans 3 --image_size 32 --num_classes 10 --model 'ADM-32' \
          --lr 1e-4 --betas 0.9 0.999 --dropout 0.0 --drop_label_prob 0.0 --total_steps 200000 --batch_size 128 \
          --beta_schedule 'linear' --loss_type 'MAPPED_MSE' --weight_type'constant' \
          --warmup_steps 5000 --cosine_decay True --class_cond True --parallel True --use_fp16 False \
          --sample_timesteps 10 --guidance_scale 1.5 --eps_scaler 1.0 --save_step 50000 --eval_step 50000 --fid_cache '' \

# consine schedule + mapped mse loss
torchrun  --nproc_per_node=2 main.py --train True --eval True --data_dir '' --dataset 'CIFAR-10' \
          --in_chans 3 --image_size 32 --num_classes 10 --model 'ADM-32' \
          --lr 1e-4 --betas 0.9 0.999 --dropout 0.0 --drop_label_prob 0.0 --total_steps 200000 --batch_size 128 \
          ---beta_schedule 'cosine' --loss_type 'MAPPED_MSE' --weight_type'constant' \
          --warmup_steps 5000 --cosine_decay True --class_cond True --parallel True --use_fp16 False \
          --sample_timesteps 10 --guidance_scale 1.5 --eps_scaler 1.0 --save_step 50000 --eval_step 50000 --fid_cache '' \

# optim schedule + mapped mse loss
torchrun  --nproc_per_node=2 main.py --train True --eval True --data_dir '' --dataset 'CIFAR-10' \
          --in_chans 3 --image_size 32 --num_classes 10 --model 'ADM-32' \
          --lr 1e-4 --betas 0.9 0.999 --dropout 0.0 --drop_label_prob 0.0 --total_steps 200000 --batch_size 128 \
          --beta_schedule 'optim_2' --loss_type 'MAPPED_MSE' --weight_type'constant' \
          --warmup_steps 5000 --cosine_decay True --class_cond True --parallel True --use_fp16 False \
          --sample_timesteps 10 --guidance_scale 1.5 --eps_scaler 1.0 --save_step 50000 --eval_step 50000 --fid_cache '' \
          
################################################################################################################################
# Unsupervised generation in CelebA
################################################################################################################################

# CelebA 64×64×3 UNet
torchrun  --nproc_per_node=2 main.py --train True --eval True --data_dir '' --dataset 'CelebA' \
          --in_chans 3 --image_size 64 --num_classes 0 --model 'UNet-64' \
          --lr 1e-4 --betas 0.9 0.999 --dropout 0.0 --drop_label_prob 0.0 --total_steps 500000 --batch_size 128 \
          --warmup_steps 5000 --cosine_decay False --class_cond False --parallel True --use_fp16 False \
          --sample_timesteps 20 --guidance_scale 1.0 --eps_scaler 1.0 --save_step 100000 --eval_step 100000 --fid_cache '' \

# CelebA 64×64×3 ViT
torchrun  --nproc_per_node=2 main.py --train True --eval True --data_dir '' --dataset 'CelebA' \
          --patch_size 4 --in_chans 3 --image_size 64 --num_classes 0 --model 'ViT-S' \
          --lr 1e-4 --betas 0.99 0.99 --dropout 0.0 --drop_label_prob 0.0 --total_steps 500000 --batch_size 128 \
          --warmup_steps 5000 --cosine_decay False --class_cond False --parallel True --use_fp16 False \
          --sample_timesteps 20 --guidance_scale 1.0 --eps_scaler 1.0 --save_step 100000 --eval_step 100000 --fid_cache '' \

################################################################################################################################
# Conditional generation in ImageNet
################################################################################################################################

# ImageNet 64×64×3 ViT
torchrun  --nproc_per_node=2 main.py --train True --eval True --data_dir '' --dataset 'ImageNet' \
          --patch_size 4 --in_chans 3 --image_size 64 --num_classes 1000 --model 'ViT-L' \
          --lr 1e-4 --betas 0.99 0.99 --dropout 0.0 --drop_label_prob 0.15 --total_steps 800000 --batch_size 256 \
          --warmup_steps 5000 --cosine_decay False --class_cond True --parallel True --use_fp16 True  \
          --sample_timesteps 20 --guidance_scale 1.5 --eps_scaler 1.0 --save_step 200000 --eval_step 200000 --fid_cache '' \

# ImageNet 32×32×4 UNet
torchrun  --nproc_per_node=2 main.py --train True --eval True --data_dir '' --dataset 'Encoded_ImageNet' \
          --in_chans 4 --image_size 32 --num_classes 1000 --model 'LDM' \
          --lr 1e-4 --betas 0.9 0.999 --dropout 0.0 --drop_label_prob 0.15 --total_steps 800000 --batch_size 256 \
          --warmup_steps 5000 --cosine_decay False --class_cond True --parallel True --use_fp16 True  \
          --sample_timesteps 50 --guidance_scale 1.5 --eps_scaler 1.0 --save_step 200000 --eval_step 200000 --fid_cache '' \

# ImageNet 32×32×4 ViT
torchrun  --nproc_per_node=2 main.py --train True --eval True --data_dir '' --dataset 'Encoded_ImageNet' \
          --patch_size 2 --in_chans 4 --image_size 32 --num_classes 1000 --model 'ViT-L' \
          --lr 1e-4 --betas 0.99 0.99 --dropout 0.0 --drop_label_prob 0.15 --total_steps 800000 --batch_size 256 \
          --warmup_steps 5000 --cosine_decay False --class_cond True --parallel True --use_fp16 True  \
          --sample_timesteps 50 --guidance_scale 1.5 --eps_scaler 1.0 --save_step 200000 --eval_step 200000 --fid_cache '' \

