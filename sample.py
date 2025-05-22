import argparse
import os
import torch
from PIL import Image
from tools.utils import *
from tools import dist_util
import tensorflow.compat.v1 as tf  # type: ignore
from tools.sampler import Sampler, Classifier
from torch.nn.parallel import DistributedDataParallel as DDP
from main import build_diffusion, build_model
from models.unet import *; from models.dit import *; from models.vit import *; from models.uvit import *


model_variants = [
    "UNet-32","ADM-32", "ADM-64", "ADM-128", "ADM-256", "ADM-512", "UNet-64", "LDM",
    "ViT-S", "ViT-B", "ViT-L", "ViT-XL",
    "DiT-S", "DiT-B", "DiT-L", "DiT-XL",
    "U-ViT-S", "U-ViT-S-D", "U-ViT-M", "U-ViT-L", "U-ViT-H"]

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate guided diffusion models")
    # model  configuration
    parser.add_argument("--model", type=str, default="DiT-XL", choices=model_variants, help="Model variant to use")    
    parser.add_argument("--patch_size", type=int, default=2, help="Patch Size for ViT, DiT, U-ViT, type is int")
    parser.add_argument("--in_chans", type=int, default=4, help="Number of input channels for the model")
    parser.add_argument("--image_size", type=int, default=32, help="Image size")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of classes, type is int")
    parser.add_argument("--class_cond", default=True, type=str2bool, help="Set class_cond to enable class-conditional generation.")    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Gaussian Diffusion
    parser.add_argument("--beta_schedule", type=str, default='cosine', help="Beta schedule type 'linear', 'cosine', 'laplace', and 'power'.")
    parser.add_argument("--p", type=float, default=2, help="power for power schedule.")
    parser.add_argument("--diffusion_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--mean_type", type=str, default='EPSILON', choices=['PREVIOUS_X', 'START_X', 'EPSILON', 'VELOCITY'], help="Predict variable")
    parser.add_argument("--var_type", type=str, default='FIXED_LARGE', choices=['FIXED_LARGE', 'FIXED_SMALL', 'LEARNED', 'LEARNED_RANGE'], help="Variance type")
    parser.add_argument("--learn_sigma", default=False, type=str2bool, help="Set learn_sigma to enable learn distribution sigma.")    

    # 
    parser.add_argument("--latent_scale", type=float, default=0.18215, help="scaling factor for latent sample normalization. (0.18215 for unit variance)")
    parser.add_argument("--parallel", default=True, type=str2bool, help="Use multi-GPU sampling")
    parser.add_argument('--amp', default=True, type=str2bool, help='Use AMP for mixed precision sampling')
    parser.add_argument('--resume', type=str, default=None, help='Path to the checkpoint to resume from')
    parser.add_argument("--save_path", type=str, default='./sample_images', help="Log directory")
    
    # Logging & Sampling
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--solver", type=str, default='heun', choices=['ddim', 'heun', 'euler'], help="Choose sampler 'ddim', 'euler' or 'heun'")
    parser.add_argument('--discretization', type=str, default='edm', choices=['vp', 've', 'iddpm', 'edm'], help='Discretization method for edm solver.')
    parser.add_argument('--schedule', type=str, default='linear', choices=['vp', 've', 'linear'], help='Noise schedule for edm sampling.')
    parser.add_argument('--scaling', type=str, default='none', choices=['vp', 'none'], help='Scaling strategy for model output in edm.')
    parser.add_argument("--sample_steps", type=int, default=18, help="Number of sample diffusion steps")
    parser.add_argument("--class_labels", type=int, nargs="+", default=[207, 360, 387, 974, 88, 979, 417, 279], help="Specify the class labels used for sampling, e.g., --class_labels 207 360 387")
    parser.add_argument("--num_samples", type=int, default=50000, help="The number of generated images for evaluation")
    parser.add_argument("--sample_size", type=int, default=64, help="Sampling size of images")
    parser.add_argument("--use_classifier", type=str, default=None, help="Path to the pre-trained classifier model")
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='Scale factor for classifier-free guidance')
    parser.add_argument('--t_from', type=int, default=-1, help='Starting timestep for finite interval guidance (non-negative, >= 0). Set to -1 to disable interval guidance.')
    parser.add_argument('--t_to', type=int, default=-1, help='Ending timestep for finite interval guidance (must be > t_from). Set to -1 to disable interval guidance.')

    args = parser.parse_args()    
    return args

def main():
    args = parse_args()
    if args.parallel:
        dist_util.setup_dist()  
        local_rank = int(os.getenv('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    set_random_seed(args, args.seed)
    
    sample_diffusion = build_diffusion(args, use_ddim=True)
    ema_model = build_model(args).to(device)
    
    if args.parallel:
        ema_model = DDP(ema_model, device_ids=[local_rank], output_device=local_rank)

    assert os.path.exists(args.resume), 'Error: checkpoint {} not found'.format(args.resume)
    checkpoint = torch.load(args.resume)
    ema_model.load_state_dict(checkpoint['ema_model'])

    classifier = Classifier(args, device, ema_model) if args.use_classifier else None
    sampler = Sampler(args, device, ema_model, sample_diffusion, classifier=classifier)
    
    with torch.no_grad():
        all_samples, _ = sampler.sample(
            num_samples=args.num_samples, 
            sample_size=args.sample_size, 
            image_size=args.image_size, 
            num_classes=args.num_classes, 
            progress_bar=True)

    if dist_util.is_main_process():
        os.makedirs(args.save_path, exist_ok=True)
        for i in range(all_samples.shape[0]):
            img = Image.fromarray(all_samples[i].numpy())  # Convert to PIL Image
            img.save(os.path.join(args.save_path, f'sample_{i:06d}.png'))

if __name__ == "__main__":
    main()
