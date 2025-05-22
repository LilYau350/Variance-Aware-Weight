#!/bin/bash

# ImageNet 32×32×4 DiT 
CUDA_VISIBLE_DEVICES=0,1 torchrun --master-port=29601 --nproc_per_node=2 main.py --train True --eval True --data_dir './ImageNet/ImageNet_256/ImageNet.h5' --dataset 'Latent' \
          --patch_size 2 --in_chans 4 --image_size 32 --num_classes 1000 --model 'DiT-S' --mean_type 'EPSILON' \
          --lr 1e-4 --betas 0.9 0.95 --dropout 0.0 --drop_label_prob 0.0 --total_steps 400000 --batch_size 256 --grad_accumulation 1 \
          --beta_schedule 'cosine' --loss_type 'MSE' --weight_type 'constant' --sampler_type 'uniform' --gamma 0.1 \
          --warmup_steps 0 --cosine_decay False --class_cond True --parallel True --amp True --sample_size 16 \
          --sample_steps 50 --guidance_scale 1.0 --sample_freq 10000 --num_samples 50000 --save_step 0 --eval_step 50000 \
          --ref_batch './preprocessing/reference_batches/VIRTUAL_imagenet256_labeled.npz'   
