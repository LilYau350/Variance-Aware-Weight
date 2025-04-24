import argparse
import csv
import os
import re
import copy
import math
import random
import warnings
import torch
import numpy as np
from tqdm import trange
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torchvision.utils import make_grid, save_image
from tools.utils import *
from tools import dist_util, logger
from evaluations.evaluator import Evaluator
import tensorflow.compat.v1 as tf  # type: ignore
from tools.trainer import Trainer
from tools.sampler import Sampler, Classifier
from datasets.data_loader import load_dataset
from tools.respace import SpacedDiffusion, space_timesteps
from torch.nn.parallel import DistributedDataParallel as DDP
from models.unet import *; from models.dit import *; from models.vit import *; from models.uvit import *
from tools.gaussian_diffusion import (
    get_named_beta_schedule,
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType
)

warnings.filterwarnings("ignore")


model_variants = [
    "UNet-32","ADM-32", "ADM-64", "ADM-128", "ADM-256", "ADM-512", "UNet-64", "LDM",
    "ViT-S", "ViT-B", "ViT-L", "ViT-XL",
    "DiT-S", "DiT-B", "DiT-L", "DiT-XL",
    "U-ViT-S", "U-ViT-S-D", "U-ViT-M", "U-ViT-L", "U-ViT-H"]

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate guided diffusion models")
    # model configuration
    parser.add_argument("--model", type=str, default="DiT-XL", choices=model_variants, help="Model variant to use")
    parser.add_argument("--patch_size", type=int, default=2, help="Patch Size for ViT, DiT, U-ViT, type is int")
    parser.add_argument("--in_chans", type=int, default=4, help="Number of input channels for the model")
    parser.add_argument("--image_size", type=int, default=32, help="Image size")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of classes, type is int")
    parser.add_argument("--class_cond", default=False, type=str2bool, help="Set class_cond to enable class-conditional generation.")
    parser.add_argument("--learn_sigma", default=False, type=str2bool, help="Set learn_sigma to enable learn distribution sigma.")  
  
    # Gaussian Diffusion
    parser.add_argument("--beta_schedule", type=str, default='cosine', help="Beta schedule type 'linear', 'cosine', 'laplace', and 'power'.")
    parser.add_argument("--p", type=float, default=2, help="power for power schedule.")
    parser.add_argument("--T", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--mean_type", type=str, default='EPSILON', choices=['PREVIOUS_X', 'START_X', 'EPSILON', 'VELOCITY', 'UNRAVEL'], help="Predict variable")
    parser.add_argument("--var_type", type=str, default='FIXED_LARGE', choices=['FIXED_LARGE', 'FIXED_SMALL', 'LEARNED', 'LEARNED_RANGE'], help="Variance type")

  
    # Logging & Sampling
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")  
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--solver", type=str, default='heun', choices=['ddim', 'heun', 'euler'], help="Choose sampler 'ddim', 'euler' or 'heun'")
    parser.add_argument('--discretization', type=str, default='edm', choices=['vp', 've', 'iddpm', 'edm'], help='Discretization method for edm solver.')
    parser.add_argument('--schedule', type=str, default='linear', choices=['vp', 've', 'linear'], help='Noise schedule for edm sampling.')
    parser.add_argument('--scaling', type=str, default='none', choices=['vp', 'none'], help='Scaling strategy for model output in edm.')
    parser.add_argument("--sample_timesteps", type=int, default=18, help="Number of sample diffusion steps")
    parser.add_argument("--class_labels", type=int, nargs="+", default=None, help="Specify the class labels used for sampling, e.g., --class_labels 207 360 387")
    parser.add_argument("--logdir", type=str, default='./logs', help="Log directory")
    parser.add_argument("--sample_size", type=int, default=64, help="Sampling size of images")
    parser.add_argument("--sample_step", type=int, default=10000, help="Frequency of sampling")
    parser.add_argument("--use_classifier", type=str, default=None, help="Path to the pre-trained classifier model")
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='Scale factor for classifier-free guidance')
    parser.add_argument('--t_from', type=int, default=-1, help='Starting timestep for finite interval guidance (non-negative, >= 0). Set to -1 to disable interval guidance.')
    parser.add_argument('--t_to', type=int, default=-1, help='Ending timestep for finite interval guidance (must be > t_from). Set to -1 to disable interval guidance.')
    parser.add_argument("--num_samples", type=int, default=50000, help="The number of generated images for evaluation")
    parser.add_argument("--latent_scale", type=float, default=0.18215, help="scaling factor for latent sample normalization. (0.18215 for unit variance)")
    parser.add_argument("--parallel", default=True, type=str2bool, help="Use multi-GPU sampling")
    parser.add_argument('--amp', default=False, type=str2bool, help='Use AMP for mixed precision training')
    parser.add_argument('--resume', type=str, default=None, help='Path to the checkpoint to resume from')
    args = parser.parse_args()    
    return args

