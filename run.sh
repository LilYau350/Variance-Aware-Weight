# ImageNet 32×32×4 DiT 
CUDA_VISIBLE_DEVICES=0,1 torchrun  --nproc_per_node=2 main.py --train True --eval True --data_dir './ImageNet/ImageNet_256/ImageNet.h5' --dataset 'Latent' \
          --patch_size 2 --in_chans 4 --image_size 32 --num_classes 1000 --model 'DiT-S' --mean_type 'EPSILON' \
          --lr 1e-4 --betas 0.9 0.95 --dropout 0.0 --drop_label_prob 0.0 --total_steps 400000 --batch_size 256 --grad_accumulation 1 \
          --beta_schedule 'cosine' --loss_type 'MSE' --weight_type 'constant' --sampler_type 'uniform' --mapping False \
          --warmup_steps 0 --cosine_decay False --class_cond True --parallel True --amp True  --sample_size 16 \
          --sample_timesteps 50 --guidance_scale 1.0 --sample_step 50000 --num_samples 50000 --save_step 100000 --eval_step 50000 \
          --ref_batch './preprocessing/reference_batches/VIRTUAL_imagenet256_labeled.npz' 

# note 
# In trainer.py#L76, We can modify trainer.py (autocast()) as with autocast(dtype=torch.bfloat16). BP16 would be more stable than FP16.
# https://github.com/LilYau350/Representation-Degradation-in-Diffusion-Training/blob/e38244941c7ae726fd0dad8ae2f9b37e95bea78e/tools/trainer.py#L76 

# Here, betas can be set to 0.99 0.99, or 0.9 0.999, we search lr {1e-3,5e-4,1e-4} first. 
