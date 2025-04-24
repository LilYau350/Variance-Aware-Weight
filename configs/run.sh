#!/bin/bash

# Hardware and environment settings
export CUDA_VISIBLE_DEVICES=0,1  # Specify GPUs to use
NPROC_PER_NODE=2                 # Number of processes per node (matches GPU count)

# Data file paths
DATA_DIR="/home/workspace/ddpm/ImageNet/ImageNet.h5"                # Absolute path to dataset
REF_BATCH="/home/workspace/ddpm/reference_batches/VIRTUAL_imagenet256_labeled.npz"  # Absolute path to reference batch

# Check if an experiment name is provided
if [ -z "$1" ]; then
    echo "Error: Please provide an experiment name"
    echo "Usage: $0 <experiment_name>"
    echo "Available experiments:"
    echo "  - dit_epsilon_baseline"
    echo "  - dit_epsilon_min_snr"
    echo "  - dit_epsilon_p2"
    echo "  - dit_epsilon_lambda"
    echo "  - dit_epsilon_adaptive"
    echo "  - dit_start_x_baseline"
    echo "  - dit_start_x_min_snr"
    echo "  - dit_start_x_adaptive"
    echo "  - dit_velocity_baseline"
    echo "  - dit_velocity_min_snr"
    echo "  - dit_velocity_adaptive"
    exit 1
fi

# Experiment name
EXPERIMENT=$1

# Set working directory to ensure consistent path resolution
cd /home/workspace/ddpm # Set working directory to where main.py is located

# Base command (without CUDA_VISIBLE_DEVICES, nproc_per_node, data_dir, and ref_batch)
BASE_CMD="torchrun --nproc_per_node=$NPROC_PER_NODE main.py --train True --eval True --dataset 'Latent' \
          --patch_size 2 --in_chans 4 --image_size 32 --num_classes 1000 --model 'DiT-S' \
          --lr 1e-4 --betas 0.9 0.999 --dropout 0.0 --drop_label_prob 0.0 --total_steps 400000 --batch_size 256 --grad_accumulation 1 \
          --beta_schedule 'linear' --loss_type 'MSE' --sampler_type 'uniform' --mapping False \
          --warmup_steps 0 --cosine_decay False --class_cond True --parallel True --amp True --sample_size 16 \
          --sample_timesteps 50 --guidance_scale 1.0 --sample_step 10000 --num_samples 50000 --save_step 100000 --eval_step 50000 \
          --data_dir '$DATA_DIR' --ref_batch '$REF_BATCH'"

# Set parameters based on experiment name
case $EXPERIMENT in
    "dit_epsilon_baseline")
        CMD="$BASE_CMD --mean_type 'EPSILON' --weight_type 'constant'"
        ;;
    "dit_epsilon_min_snr")
        CMD="$BASE_CMD --mean_type 'EPSILON' --weight_type 'min_snr_5'"
        ;;
    "dit_epsilon_p2")
        CMD="$BASE_CMD --mean_type 'EPSILON' --weight_type 'p2'"
        ;;
    "dit_epsilon_lambda")
        CMD="$BASE_CMD --mean_type 'EPSILON' --weight_type 'lambda'"
        ;;
    "dit_epsilon_adaptive")
        CMD="$BASE_CMD --mean_type 'EPSILON' --weight_type 'adaptive'"
        ;;
    "dit_start_x_baseline")
        CMD="$BASE_CMD --mean_type 'START_X' --weight_type 'constant'"
        ;;
    "dit_start_x_min_snr")
        CMD="$BASE_CMD --mean_type 'START_X' --weight_type 'min_snr_5'"
        ;;
    "dit_start_x_adaptive")
        CMD="$BASE_CMD --mean_type 'START_X' --weight_type 'adaptive'"
        ;;
    "dit_velocity_baseline")
        CMD="$BASE_CMD --mean_type 'VELOCITY' --weight_type 'constant'"
        ;;
    "dit_velocity_min_snr")
        CMD="$BASE_CMD --mean_type 'VELOCITY' --weight_type 'vmin_snr_5'"
        ;;
    "dit_velocity_adaptive")
        CMD="$BASE_CMD --mean_type 'VELOCITY' --weight_type 'adaptive'"
        ;;
    *)
        echo "Error: Unknown experiment name '$EXPERIMENT'"
        echo "Available experiments:"
        echo "  - dit_epsilon_baseline"
        echo "  - dit_epsilon_min_snr"
        echo "  - dit_epsilon_p2"
        echo "  - dit_epsilon_lambda"
        echo "  - dit_epsilon_adaptive"
        echo "  - dit_start_x_baseline"
        echo "  - dit_start_x_min_snr"
        echo "  - dit_start_x_adaptive"
        echo "  - dit_velocity_baseline"
        echo "  - dit_velocity_min_snr"
        echo "  - dit_velocity_adaptive"
        exit 1
        ;;
esac

# Execute command
echo "Running experiment: $EXPERIMENT"
echo "Command: $CMD"
eval $CMD
