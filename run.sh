#!/bin/bash

################################################################################################################################
# Unsupervised generation in CelebA
################################################################################################################################

# # CelebA 64×64×3 UNet
# torchrun  --nproc_per_node=2 main.py --train True --eval True --data_dir '' --dataset 'CelebA' \
#           --in_chans 3 --image_size 64 --num_classes 0 --model 'UNet-64' \
#           --lr 1e-4 --betas 0.9 0.999 --dropout 0.0 --drop_label_prob 0.0 --total_steps 500000 --batch_size 128 \
#           --beta_schedule 'cosine' --loss_type 'MSE' --weight_type 'lambda' --sampler_type 'uniform'\
#           --warmup_steps 5000 --cosine_decay False --class_cond False --parallel True --amp True \
#           --sample_timesteps 20 --guidance_scale 1.0 --eps_scaler 1.0 --save_step 100000 --eval_step 100000 --fid_cache '' \

# CelebA 64×64×3 ViT
torchrun  --nproc_per_node=2 main.py --train True --eval True --data_dir '' --dataset 'CelebA' \
          --patch_size 4 --in_chans 3 --image_size 64 --num_classes 0 --model 'ViT-S' \
          --lr 1e-3 --betas 0.99 0.99 --dropout 0.0 --drop_label_prob 0.0 --total_steps 500000 --batch_size 128 \
          --beta_schedule 'cosine' --loss_type 'MSE' --weight_type 'lambda' --sampler_type 'uniform'\
          --warmup_steps 5000 --cosine_decay False --class_cond False --parallel True --amp True \
          --sample_timesteps 20 --guidance_scale 1.0 --eps_scaler 1.0 --save_step 100000 --eval_step 100000 --fid_cache '' \

# CelebA 64×64×3 DiT
torchrun  --nproc_per_node=2 main.py --train True --eval True --data_dir '' --dataset 'CelebA' \
          --patch_size 4 --in_chans 3 --image_size 64 --num_classes 1 --model 'DiT-S' \
          --lr 1e-3 --betas 0.9 0.999 --dropout 0.0 --drop_label_prob 0.0 --total_steps 500000 --batch_size 128 \
          --beta_schedule 'cosine' --loss_type 'MSE' --weight_type 'lambda' --sampler_type 'uniform'\
          --warmup_steps 5000 --cosine_decay False --class_cond True --parallel True --amp True \
          --sample_timesteps 20 --guidance_scale 1.0 --eps_scaler 1.0 --save_step 100000 --eval_step 100000 --fid_cache '' \

################################################################################################################################
# Conditional generation in ImageNet
################################################################################################################################

# # ImageNet 64×64×3 ViT
# torchrun  --nproc_per_node=2 main.py --train True --eval True --data_dir '' --dataset 'ImageNet' \
#           --patch_size 4 --in_chans 3 --image_size 64 --num_classes 1000 --model 'DiT-L' \
#           --lr 1e-4 --betas 0.99 0.99 --dropout 0.0 --drop_label_prob 0.15 --total_steps 800000 --batch_size 256 \
#           --beta_schedule 'cosine' --loss_type 'MSE' --weight_type 'constant' --sampler_type 'uniform' --mapping True \    
#           --warmup_steps 5000 --cosine_decay False --class_cond True --parallel True --amp True  \
#           --sample_timesteps 20 --guidance_scale 1.5 --eps_scaler 1.0 --save_step 200000 --eval_step 200000 --fid_cache '' \

# # ImageNet 32×32×4 UNet
# torchrun  --nproc_per_node=2 main.py --train True --eval True --data_dir '' --dataset 'Latent' \
#           --in_chans 4 --image_size 32 --num_classes 1000 --model 'LDM' \
#           --lr 1e-4 --betas 0.9 0.999 --dropout 0.0 --drop_label_prob 0.15 --total_steps 800000 --batch_size 256 \
#           --beta_schedule 'cosine' --loss_type 'MSE' --weight_type 'constant' --sampler_type 'uniform' --mapping True \
#           --warmup_steps 5000 --cosine_decay False --class_cond True --parallel True --amp True  \
#           --sample_timesteps 50 --guidance_scale 1.5 --eps_scaler 1.0 --save_step 200000 --eval_step 200000 --fid_cache '' \

# ImageNet 32×32×4 ViT
torchrun  --nproc_per_node=2 main.py --train True --eval True --data_dir '' --dataset 'Latent' \
          --patch_size 2 --in_chans 4 --image_size 32 --num_classes 1000 --model 'DiT-B' \
          --lr 1e-4 --betas 0.99 0.99 --dropout 0.0 --drop_label_prob 0.15 --total_steps 800000 --batch_size 256 \
          --beta_schedule 'cosine' --loss_type 'MSE' --weight_type 'constant' --sampler_type 'uniform' --mapping True \
          --warmup_steps 5000 --cosine_decay False --class_cond True --parallel True --amp True  \
          --sample_timesteps 50 --guidance_scale 1.5 --eps_scaler 1.0 --save_step 200000 --eval_step 200000 --fid_cache '' \

