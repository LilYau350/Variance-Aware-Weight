#!/bin/bash

# ImageNet 32×32×4 DiT 
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master-port=12345 --nproc_per_node=4 main.py --train True --eval True --data_dir '/data/ImageNet/ImageNet.h5' --dataset 'Latent' \
          --patch_size 2 --in_chans 4 --image_size 32 --num_classes 1000 --model 'DiT-B' --mean_type 'EPSILON' \
          --lr 1e-4 --betas 0.9 0.95 --dropout 0.0 --drop_label_prob 0.0 --total_steps 400000 --batch_size 256 --grad_accumulation 1 \
          --path_type 'cosine' --loss_type 'MSE' --weight_type 'lambda' --sampler_type 'ode' --solver 'heun' --gamma 0.0 \
          --warmup_steps 0 --cosine_decay False --class_cond True --parallel True --amp True --sample_size 16 \
          --sample_steps 50 --guidance_scale 1.0 --sample_freq 5000 --num_samples 50000 --save_step 100000 --eval_step 100000 \
          --ref_batch './reference_batches/VIRTUAL_imagenet256_labeled.npz'   

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master-port=12345 --nproc_per_node=4 main.py --train True --eval True --data_dir '/data/ImageNet/ImageNet.h5' --dataset 'Latent' \
          --patch_size 2 --in_chans 4 --image_size 32 --num_classes 1000 --model 'DiT-L' --mean_type 'EPSILON' \
          --lr 1e-4 --betas 0.9 0.95 --dropout 0.0 --drop_label_prob 0.0 --total_steps 400000 --batch_size 256 --grad_accumulation 1 \
          --path_type 'cosine' --loss_type 'MSE' --weight_type 'lambda' --sampler_type 'ode' --solver 'heun' --gamma 0.0 \
          --warmup_steps 0 --cosine_decay False --class_cond True --parallel True --amp True --sample_size 16 \
          --sample_steps 50 --guidance_scale 1.0 --sample_freq 5000 --num_samples 50000 --save_step 100000 --eval_step 100000 \
          --ref_batch './reference_batches/VIRTUAL_imagenet256_labeled.npz' 

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master-port=12345 --nproc_per_node=4 main.py --train True --eval True --data_dir '/data/ImageNet/ImageNet.h5' --dataset 'Latent' \
          --patch_size 2 --in_chans 4 --image_size 32 --num_classes 1000 --model 'DiT-XL' --mean_type 'EPSILON' \
          --lr 1e-4 --betas 0.9 0.95 --dropout 0.0 --drop_label_prob 0.0 --total_steps 400000 --batch_size 256 --grad_accumulation 1 \
          --path_type 'cosine' --loss_type 'MSE' --weight_type 'lambda' --sampler_type 'ode' --solver 'heun' --gamma 0.0 \
          --warmup_steps 0 --cosine_decay False --class_cond True --parallel True --amp True --sample_size 16 \
          --sample_steps 50 --guidance_scale 1.0 --sample_freq 5000 --num_samples 50000 --save_step 100000 --eval_step 100000 \
          --ref_batch './reference_batches/VIRTUAL_imagenet256_labeled.npz' 
