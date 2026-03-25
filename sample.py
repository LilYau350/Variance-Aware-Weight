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
    # Enable/Disable Training and Evaluation
    parser.add_argument("--train", default=False, type=str2bool, help="Enable training")
    parser.add_argument("--eval", default=True, type=str2bool, help="Load checkpoint and evaluate FID...")

    parser.add_argument("--data_dir", type=str, default='./data', help="Path to the dataset directory")
    parser.add_argument("--dataset", type=str, default='CIFAR-10', choices=['CIFAR-10', 'Gaussian', 'CelebA', 'ImageNet', 'LSUN', 'Latent', 'Latent_Pixel'], help="Dataset to train on")
    parser.add_argument("--patch_size", type=int, default=None, help="Patch Size for ViT, DiT, U-ViT, type is int")
    parser.add_argument("--in_chans", type=int, default=3, help="Number of input channels for the model")
    parser.add_argument("--image_size", type=int, default=32, help="Image size")
    parser.add_argument("--num_classes", type=int, default=0, help="Number of classes, type is int")
    parser.add_argument("--model", type=str, default="ADM-32", choices=model_variants, help="Model variant to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")


    # Gaussian Diffusion
    parser.add_argument("--model_mode",type=str,default="diffusion",choices=["diffusion", "flow"],
                                                    help="Choose diffusion mode: 'flow' for SDE/ODE-based modeling, 'diffusion' for DDPM-like modeling.")
    parser.add_argument("--path_type", type=str, default='linear', choices=['linear', 'cosine'], help="Path type for flow matching and diffusion")  
    parser.add_argument('--time_dist', nargs='+', default=['uniform', -0.8, 0.8], help="Time sampling distribution for mean flow training: ['uniform'] or ['lognorm', mu, sigma]")
    
    # Flow matching
    parser.add_argument('--sampler_type', type=str, default='sde', choices=['sde', 'ode'], help='Type of flow matching sampler to use')   
    # Discrete Diffusion
    parser.add_argument("--diffusion_steps", type=int, default=1000, help="Number of diffusion steps")
    # loss type for diffusion or flow matching
    parser.add_argument("--mean_type", type=str, default='EPSILON', choices=['PREVIOUS_X', 'START_X', 'EPSILON', 'VELOCITY', 'VECTOR', 'SCORE'], help="Predict variable")
    parser.add_argument("--var_type", type=str, default='FIXED_LARGE', choices=['FIXED_LARGE', 'FIXED_SMALL', 'LEARNED', 'LEARNED_RANGE'], help="Variance type")
    parser.add_argument("--loss_type", type=str, default='MSE', choices=['MSE', 'RESCALED_MSE', 'KL', 'RESCALED_KL'], help="Loss type")
    parser.add_argument("--weight_type", type=str, default='constant', help="'constant', 'lambda', 'min_snr_k','vmin_snr_k', 'max_snr_k' 'debias', where k is a positive integer.")
    parser.add_argument("--gamma", type=float, default=0, help="Coefficient for loss regularization, e.g., projection loss")
    parser.add_argument("--p2_gamma", type=int, default=1, help="hyperparameter for P2 weight")
    parser.add_argument("--p2_k", type=int, default=1, help="hyperparameter for P2 weight")
    
    parser.add_argument("--enc-type", type=str, default="dinov2-vit-b", help="Encoder specification, e.g. 'dinov2-vit-b' or comma-separated list")
    parser.add_argument("--encoder-depth", type=int, default=0, help="How many encoder blocks from SiT to expose to z-projection")

    # Training
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for DataLoader")    
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")    
    parser.add_argument("--total_steps", type=int, default=400000, help="Total training steps") 
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")        
    parser.add_argument("--class_cond", default=False, type=str2bool, help="Set class_cond to enable class-conditional generation.")
    parser.add_argument("--learn_sigma", default=False, type=str2bool, help="Set learn_sigma to enable learn distribution sigma.")   
    parser.add_argument("--learn_align", default=False, type=str2bool, help="Set learn_align to make representation align.")  
    # Adam settings
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999), help='Beta values for optimization')
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for the optimizer")
    parser.add_argument("--eps", type=float, default=1e-8, help="eps for the optimizer")
    # CFG training
    parser.add_argument('--drop_label_prob', type=float, default=0.0, help='Probability of dropping labels for classifier-free guidance')    
    # Sampling latnet
    parser.add_argument("--latent_scale", type=float, default=0.18215, help="scaling factor for latent sample normalization. (0.18215 for unit variance)")
    # Training tircks
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Learning rate warmup")    
    parser.add_argument("--final_lr", type=float, default=0.0, help="Final learning rate")
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient norm clipping")
    parser.add_argument("--dropout", type=float, default=0.0, help='Dropout rate of resblock')
    parser.add_argument("--cosine_decay", default=True, type=str2bool, help="Whether to use cosine learning rate decay")
    # DDP nad mixed precision training
    parser.add_argument("--parallel", default=False, type=str2bool, help="Use multi-GPU training")
    parser.add_argument('--amp', default=True, type=str2bool, help='Use AMP for mixed precision training')
    parser.add_argument('--grad_accumulation', type=int, default=1, help='Number of gradient accumulation steps (default: 1, no accumulation)')
    parser.add_argument('--resume', type=str, default=None, help='Path to the checkpoint to resume from')   
    parser.add_argument("--save_path", type=str, default='./sample_images', help="Log directory")

    # Logging & Sampling
    parser.add_argument("--logdir", type=str, default='./logs', help="Log directory")
    parser.add_argument("--sample_size", type=int, default=64, help="Sampling batch size of images")
    parser.add_argument("--sample_freq", type=int, default=10000, help="Frequency of sampling during training")        
    parser.add_argument("--sample_steps", type=int, default=18, help="Number of sample diffusion steps")   
    parser.add_argument("--class_labels", type=int, nargs="+", default=None, help="Specify the class labels used for sampling, e.g., --class_labels 207 360 387") 
    parser.add_argument("--use_classifier", type=str, default=None, help="Path to the pre-trained classifier model")
    # cfg and limited interval guidance
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='Scale factor for classifier-free guidance')        
    parser.add_argument('--interval', type=float, nargs=2, default=[-1.0, -1.0], metavar=('t_from', 't_to'), help='Finite interval guidance. Use -1 -1 to disable.')  
    # which version of latnet model to choose
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    # ode/sde solver
    parser.add_argument("--solver", type=str, default='heun', help="Choose sampler 'ddim', 'euler' or 'heun', 'heun2', 'dopri5'")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    # edm sampler
    parser.add_argument('--discretization', type=str, default='edm', choices=['vp', 've', 'iddpm', 'edm'], help='Discretization method for edm solver.')
    parser.add_argument('--schedule', type=str, default='linear', choices=['vp', 've', 'linear'], help='Noise schedule for edm sampling.')
    parser.add_argument('--scaling', type=str, default='none', choices=['vp', 'none'], help='Scaling strategy for model output in edm.')


    # Evaluation
    parser.add_argument("--save_step", type=int, default=100000, help="Frequency of saving checkpoints, 0 to disable during training")
    parser.add_argument("--eval_step", type=int, default=50000, help="Frequency of evaluating model, 0 to disable during training")
    parser.add_argument("--num_samples", type=int, default=50000, help="The number of generated images for evaluation")
    parser.add_argument("--ref_batch", type=str, default='./reference_batches/fid_stats_cifar_train.npz', help="FID cache")

    args = parser.parse_args()    
    return args


def main():
    args = parse_args()
    if args.parallel:
        dist_util.setup_dist()
        local_rank = int(os.getenv('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
    else:
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_random_seed(args, args.seed)

    sample_diffusion = build_diffusion(args, device, use_ddim=True)

    ema_model = build_model(args).to(device)

    if args.parallel:
        ema_model = DDP(ema_model, device_ids=[local_rank], output_device=local_rank)

    assert args.resume is not None, "--resume must be provided"
    assert os.path.exists(args.resume), f"Error: checkpoint {args.resume} not found"
    load_checkpoint(args.resume, ema_model=ema_model)

    classifier = Classifier(args, device, ema_model) if args.use_classifier else None
    sampler = Sampler(args, device, ema_model, sample_diffusion, classifier=classifier)

    with torch.no_grad():
        all_samples, all_labels = sampler.sample(
            num_samples=args.num_samples,
            sample_size=args.sample_size,
            image_size=args.image_size,
            num_classes=args.num_classes,
            progress_bar=True
        )

    if dist_util.is_main_process():
        os.makedirs(args.save_path, exist_ok=True)

        all_samples = np.concatenate(all_samples, axis=0)[:args.num_samples]

        if all_labels is not None and len(all_labels) > 0:
            all_labels = np.concatenate(all_labels, axis=0)[:args.num_samples]
        else:
            all_labels = None

        label_counters = {}

        for i in range(all_samples.shape[0]):
            img = Image.fromarray(all_samples[i])

            if all_labels is not None:
                label = int(all_labels[i])
                class_dir = os.path.join(args.save_path, str(label))
                os.makedirs(class_dir, exist_ok=True)

                if label not in label_counters:
                    label_counters[label] = 0

                save_idx = label_counters[label]
                img.save(os.path.join(class_dir, f'sample_{save_idx:06d}.png'))
                label_counters[label] += 1
            else:
                img.save(os.path.join(args.save_path, f'sample_{i:06d}.png'))

    if args.parallel:
        dist.barrier()
        dist_util.cleanup_dist()
        
if __name__ == "__main__":
    main()
