import argparse
import csv
import os
import re
import copy
import math
import time
import json
import random
import warnings
import torch
from tqdm import trange, tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
from functools import partial
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DistributedSampler
from torchvision.utils import make_grid, save_image
from tools import dist_util, logger
from evaluations.evaluator import Evaluator
import tensorflow.compat.v1 as tf # type: ignore
from datasets.data_loader import load_dataset
from tools.cfg_edm import ablation_sampler, float_equal, Net
from models.unet import *; from models.dit import *; from models.vit import *; from models.uvit import *
from tools.trainer import Trainer
from diffusers.models import AutoencoderKL
from tools.resample import LossAwareSampler, UniformSampler, create_named_schedule_sampler
from tools.respace import SpacedDiffusion, space_timesteps
from tools.gaussian_diffusion import get_named_beta_schedule, GaussianDiffusion, ModelMeanType, ModelVarType, LossType
warnings.filterwarnings("ignore")

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

model_variants = [
    "UNet-32","ADM-32", "ADM-64", "ADM-128", "ADM-256", "ADM-512", "UNet-64", "LDM",
    "ViT-S", "ViT-B", "ViT-L", "ViT-XL",
    "DiT-S", "DiT-B", "DiT-L", "DiT-XL",
    "UViT-S", "UViT-S-D", "UViT-M", "UViT-L", "UViT-H"
]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
     
def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate guided diffusion models")
    # Enable/Disable Training and Evaluation
    parser.add_argument("--train", default=True, type=str2bool, help="Enable training")
    parser.add_argument("--eval", default=True, type=str2bool, help="Load checkpoint and evaluate FID...")
    parser.add_argument("--data_dir", type=str, default='./data', help="Path to the dataset directory")
    parser.add_argument("--dataset", type=str, default='CIFAR-10', choices=['CIFAR-10', 'CelebA', 'ImageNet', 'LSUN', 'Encoded_ImageNet'], help="Dataset to train on")
    parser.add_argument("--patch_size", type=int, default=None, help="Patch Size for ViT, DiT, U-ViT, type is int")
    parser.add_argument("--in_chans", type=int, default=3, help="Number of input channels for the model")
    parser.add_argument("--image_size", type=int, default=32, help="Image size")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes, type is int")
    parser.add_argument("--model", type=str, default="ADM-32", choices=model_variants, help="Model variant to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Gaussian Diffusion
    parser.add_argument("--beta_schedule", type=str, default='optim_2', help="Beta schedule type: 'linear', 'cosine', or 'optim_k' where k is a positive integer.")
    parser.add_argument("--beta_1", type=float, default=1e-4, help="Starting value of beta for the diffusion process")
    parser.add_argument("--beta_T", type=float, default=0.2, help="Ending value of beta for the diffusion process")
    parser.add_argument("--T", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--mean_type", type=str, default='EPSILON', choices=['PREVIOUS_X', 'START_X', 'EPSILON', 'VELOCITY'], help="Predict variable")
    parser.add_argument("--var_type", type=str, default='FIXED_LARGE', choices=['FIXED_LARGE', 'FIXED_SMALL', 'LEARNED', 'LEARNED_RANGE'], help="Variance type")
    parser.add_argument("--loss_type", type=str, default='MAPPED_MSE', choices=['MAPPED_MSE', 'MIXED', 'MSE', 'L1', 'RESCALED_MSE', 'KL', 'RESCALED_KL'], help="Loss type")
    parser.add_argument("--weight_type", type=str, default='constant', help="Type of MSE loss weight: 'constant', 'min_snr_k', 'vmin_snr_k', 'max_snr_k' where k is a positive integer.")

    # Training
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999), help='Beta values for optimization')
    parser.add_argument("--final_lr", type=float, default=1e-5, help="Final learning rate")
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient norm clipping")
    parser.add_argument("--dropout", type=float, default=0.1, help='Dropout rate of resblock')
    parser.add_argument('--drop_label_prob', type=float, default=0.1, help='Probability of dropping labels for classifier-free guidance')
    parser.add_argument("--total_steps", type=int, default=400000, help="Total training steps")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Learning rate warmup")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")
    parser.add_argument('--sampler_type', type=str, default='loss-second-moment', choices=['uniform', 'loss-second-moment'], help='Type of schedule sampler to use')
    parser.add_argument("--cosine_decay", default=False, type=str2bool, help="Whether to use cosine learning rate decay")
    parser.add_argument("--class_cond", default=True, type=str2bool, help="Set class_cond to enable class-conditional generation.")
    parser.add_argument("--learn_sigma", default=False, type=str2bool, help="Set learn_sigma to enable learn distribution sigma.")
    parser.add_argument("--parallel", default=True, type=str2bool, help="Use multi-GPU training")
    parser.add_argument('--amp', default=False, type=str2bool, help='Use AMP for mixed precision training')
    parser.add_argument('--resume', type=str, default=None, help='Path to the checkpoint to resume from')
    
    # Logging & Sampling
    parser.add_argument("--sampler", type=str, default="heun", choices=["ddim", "heun"], help="Choose sampler between 'ddim' and 'heun'")
    parser.add_argument("--sample_timesteps", type=int, default=10, help="Number of sample diffusion steps")
    parser.add_argument("--logdir", type=str, default='./logs', help="Log directory")
    parser.add_argument("--sample_size", type=int, default=64, help="Sampling size of images")
    parser.add_argument("--sample_step", type=int, default=10000, help="Frequency of sampling")
    parser.add_argument("--use_classifier", type=str, default=None, help="Path to the pre-trained classifier model")
    parser.add_argument('--guidance_scale', type=float, default=1.5, help='Scale factor for classifier-free guidance')
    parser.add_argument('--eps_scaler', type=float, default=1.000, help='Scale factor for eps_scaler')
    
    # Evaluation
    parser.add_argument("--save_step", type=int, default=100000, help="Frequency of saving checkpoints, 0 to disable during training")
    parser.add_argument("--eval_step", type=int, default=50000, help="Frequency of evaluating model, 0 to disable during training")
    parser.add_argument("--num_samples", type=int, default=50000, help="The number of generated images for evaluation")
    parser.add_argument("--fid_cache", type=str, default='./stats/fid_stats_cifar_train.npz', help="FID cache")

    args = parser.parse_args()    
    return args

def save_checkpoint(model, optimizer, step, args, ema_model=None):
    if dist_util.is_main_process():
        checkpoint_dir = 'checkpoint'
        os.makedirs(checkpoint_dir, exist_ok=True)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step
        }
        if ema_model is not None:
            state['ema_model'] = ema_model.state_dict()
        filename = f"{args.loss_type}_{args.beta_schedule}"
        
        if args.beta_schedule == "optim":
            filename += f"_{args.k}"
            
        filename += f"_{step}.pth"
        filename = os.path.join(checkpoint_dir, filename)
        torch.save(state, filename)
        print(f"Checkpoint saved: {filename}")

def load_checkpoint(ckpt_path, model=None, optimizer=None, ema_model=None):
    if dist_util.is_main_process():
        print('==> Resuming from checkpoint..')
    assert os.path.exists(ckpt_path), 'Error: checkpoint {} not found'.format(ckpt_path)
    checkpoint = torch.load(ckpt_path)
    if model:
        model.load_state_dict(checkpoint['model'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if ema_model and 'ema_model' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_model'])
    return checkpoint

def create_classifier(args, model):
    attention_ds = [args.image_size // res for res in model.attention_resolutions]

    classifier_kwargs = {
        'image_size': model.image_size,
        'in_channels': args.in_chans,
        'model_channels': model.model_channels,
        'out_channels': model.num_classes,
        'num_res_blocks': model.num_res_blocks,
        'attention_resolutions': tuple(attention_ds),
        'channel_mult': model.channel_mult,
        'num_head_channels': model.num_head_channels,
        'use_scale_shift_norm': model.use_scale_shift_norm,
        'resblock_updown': model.resblock_updown,
        'pool': "attention",
    }

    return EncoderUNetModel(**classifier_kwargs)

def load_classifier(args, model, device):
    classifier = create_classifier(args, model)
    
    classifier.load_state_dict(
        dist_util.load_state_dict(args.use_classifier, map_location="cpu")
    )
    classifier.to(device)
    classifier.eval()
    
    return classifier

def build_dataset(args):
    if args.dataset == 'CIFAR-10':
        image_size = args.image_size or 32
        train_loader, test_loader, input_channels, image_size = load_dataset(
            args.data_dir, args.dataset, args.batch_size, image_size, num_workers=args.num_workers, shuffle=not args.parallel)
    elif args.dataset == 'CelebA':
        image_size = args.image_size or 64
        train_loader, test_loader, input_channels, image_size = load_dataset(
            args.data_dir, 'CelebA', args.batch_size, image_size, num_workers=args.num_workers, shuffle=not args.parallel)
    elif args.dataset == 'ImageNet':
        if args.image_size not in [64, 128, 256]:
            raise ValueError("Image size for ImageNet must be one of [64, 128, 256]")
        image_size = args.image_size
        train_loader, test_loader, input_channels, image_size = load_dataset(
            args.data_dir, args.dataset, args.batch_size, image_size, num_workers=args.num_workers, shuffle=not args.parallel)
    elif args.dataset == 'LSUN':
        image_size = args.image_size or 256
        train_loader, test_loader, input_channels, image_size = load_dataset(
            args.data_dir, args.dataset, args.batch_size, image_size, num_workers=args.num_workers, shuffle=not args.parallel)
    elif args.dataset == 'Encoded_ImageNet':
        image_size = args.image_size or 32  # Assuming encoded ImageNet is 32x32
        train_loader, test_loader, input_channels, image_size = load_dataset(
            args.data_dir, 'Encoded_ImageNet', args.batch_size, image_size, num_workers=args.num_workers, shuffle=not args.parallel)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    if args.parallel:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        per_gpu_batch_size = args.batch_size // world_size

        train_sampler = DistributedSampler(train_loader.dataset, num_replicas=world_size, rank=rank)

        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset, 
            batch_size=per_gpu_batch_size,  
            sampler=train_sampler,          
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,  
        )
        
    return train_loader, test_loader, input_channels, image_size

def build_model(args):
    unet_models = {
        "UNet-32": UNet_32,"ADM-32": ADM_32, "ADM-64": ADM_64, "ADM-128": ADM_128, 
        "ADM-256": ADM_256, "ADM-512": ADM_512, "UNet-64": UNet_64, "LDM": LDM,
    }
    vit_models = {
        "ViT-S": ViT_S, "ViT-B": ViT_B, "ViT-L": ViT_L, "ViT-XL": ViT_XL
    }
    dit_models = {
        "DiT-S": DiT_S, "DiT-B": DiT_B, "DiT-L": DiT_L, "DiT-XL": DiT_XL
    }
    uvit_models = {
        "UViT-S": UViT_S, "UViT-S-D": UViT_S_D, "UViT-M": UViT_M, 
        "UViT-L": UViT_L, "UViT-H": UViT_H
    }

    model_dict = {**unet_models, **vit_models, **dit_models, **uvit_models}

    if args.model not in model_dict:
        raise ValueError(f"Unsupported model variant: {args.model}")

    if any(x in args.model for x in ["UNet", "ADM", "LDM"]):
        model = model_dict[args.model](num_classes=args.num_classes, in_channels=args.in_chans, drop_label_prob=args.drop_label_prob, 
                                       dropout=args.dropout, learn_sigma=args.learn_sigma, class_cond=args.class_cond,
                                       )
        
    elif "ViT" in args.model:
        model = model_dict[args.model](
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        in_channels=args.in_chans,
        learn_sigma=args.learn_sigma, 
        drop_rate=args.dropout, 
        drop_label_prob=args.drop_label_prob,  
        )
    else:
        model = model_dict[args.model](
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
        )
    
    return model

def build_diffusion(args, use_ddim=False):

    betas = get_named_beta_schedule(args.beta_schedule, args.T)
    
    if use_ddim and args.sample_timesteps < args.T:  
        timestep_respacing = f"ddim{args.sample_timesteps}"  # Use DDIM and specify the number of sampling steps.
    else:
        timestep_respacing = [args.T]
    
    diffusion_kwargs = dict(
        betas=betas,
        model_mean_type=ModelMeanType[args.mean_type.upper()],
        model_var_type=ModelVarType[args.var_type.upper()],
        loss_type=LossType[args.loss_type.upper()],
        rescale_timesteps=True,
        eps_scaler=args.eps_scaler,
        mse_loss_weight_type=args.weight_type
    )

    if use_ddim:
        return SpacedDiffusion(
            use_timesteps=space_timesteps(args.T, timestep_respacing),
            **diffusion_kwargs)
    else:
        return GaussianDiffusion(**diffusion_kwargs)


def set_random_seed(args, seed):
    rank = dist.get_rank() if args.parallel else 0
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def warmup_cosine_lr(step, warmup_steps, total_steps, lr, final_lr, cosine_decay):
    if step < warmup_steps:
        return min(step, warmup_steps) / warmup_steps
    else:
        if cosine_decay:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return (final_lr + (lr - final_lr) * cosine_decay) / lr
        else:
            return 1
        
def sync_ema_model(ema_model):
    for param in ema_model.parameters():
        dist.broadcast(param.data, src=0)  # src=0 means broadcast from rank 0
        
def save_metrics_to_csv(args, eval_dir, metrics, step):
    params = (
        f"{args.dataset}_{args.model}_"
        + (f"patch_{args.patch_size}_" if args.patch_size else "")
        + f"lr_{args.lr}_"  
        + f"dropout_{args.dropout}_"
        + f"drop_label_{args.drop_label_prob}_"
        + f"sample_t_{args.sample_timesteps}_"
        + f"cfg_{args.guidance_scale}_"
        + f"beta_sched_{args.beta_schedule}_"
        + f"loss_{args.loss_type}_"
        + f"weight_{args.weight_type}_"  
        + (f"gradclip_{args.grad_clip}_" if args.grad_clip else "")
        + ("cond_" if args.class_cond else "")
    )

    params = re.sub(r'[^\w\-_\. ]', '_', params).rstrip('_')

    csv_filename = os.path.join(eval_dir, f"{params}.csv")
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Step'] + list(metrics.keys()))
        writer.writerow([step] + list(metrics.values()))
        
def cond_fn(x, t, y=None,classifier=None, classifier_scale=1.0):
    assert y is not None  
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(log_probs)), y.view(-1)]
        return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale

def model_fn(model, args, x, t, y=None):
    assert y is not None
    return model(x, t, y if args.class_cond else None)

def ddim_sampler(args, model, sample_diffusion, num_samples, sample_size, image_size, num_classes, device, progress_bar=False):
    model.eval()  
    # for param in model.parameters():
    #     param.requires_grad = False
    classifier = None
    if args.use_classifier:
        classifier = load_classifier(args, device)
    
    if args.parallel:
        sync_ema_model(model)  
        
    if args.parallel: 
        dist.barrier()  
          
    all_samples = []
    all_labels = []
    rank = dist.get_rank() if args.parallel else 0
    world_size = dist.get_world_size() if args.parallel else 1
    
    if progress_bar and dist_util.is_main_process():
        pbar = tqdm(total=num_samples, desc=f"Generating Samples", leave=True)
        
    while len(all_samples) * sample_size < num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = torch.randint(low=0, high=num_classes, size=(sample_size,), device=device)
            model_kwargs["y"] = classes

        sample = sample_diffusion.ddim_sample_loop(
            model if not args.use_classifier else model_fn,
            (sample_size, 3, image_size, image_size),
            clip_denoised=True,
            device=device,
            model_kwargs=model_kwargs,
            cond_fn=(lambda x, t, y: cond_fn(x, t, y, classifier, args.guidance_scale)) if args.use_classifier else None  
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()
        
        if args.parallel: 
            gathered_samples = [torch.zeros_like(sample) for _ in range(world_size)]
            dist.all_gather(gathered_samples, sample)
            all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

            if args.class_cond:
                gathered_labels = [torch.zeros_like(classes) for _ in range(world_size)]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        else:
            all_samples.append(sample.cpu().numpy())

            if args.class_cond:
                all_labels.append(classes.cpu().numpy())

        if dist_util.is_main_process() and progress_bar:
            pbar.update(sample_size * world_size)
            
    if dist_util.is_main_process() and progress_bar:
        pbar.close()
        
    model.train()  
       
    return all_samples, all_labels

def heun_sampler(args, model, sample_diffusion, num_samples, sample_size, image_size, num_classes, device, progress_bar=False):
    model.eval()     
    if args.parallel:
        sync_ema_model(model)  
        
    if args.parallel: 
        dist.barrier()  
          
    all_samples = []
    all_labels = []
    
    rank = dist.get_rank() if args.parallel else 0
    world_size = dist.get_world_size() if args.parallel else 1
    
    if progress_bar and dist_util.is_main_process():
        pbar = tqdm(total=num_samples, desc=f"Generating Samples", leave=True)
        
    if args.in_chans == 4:
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse").cuda()
       
    net = Net(model=model, img_channels=args.in_chans, 
            img_resolution=image_size,  
            noise_schedule=args.beta_schedule,
            amp=args.amp,).to(device)

    while len(all_samples) * sample_size < num_samples:
        class_labels = None
        if args.class_cond:
            y_cond = torch.randint(low=0, high=num_classes, size=(sample_size,), device=device)
            if not float_equal(args.guidance_scale, 1.0):
                y_uncond = torch.randint(low=num_classes, high=num_classes + 1, size=(sample_size,), device=device)
                class_labels = torch.cat((y_cond, y_uncond), dim=0)
            else:
                y_uncond = None
                class_labels = y_cond

        z = torch.randn([sample_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        
        # if args.guidance_scale > 0:
        if not float_equal(args.guidance_scale, 1.0):
            z = torch.cat((z, z), dim=0)

        #guidance_scale = args.guidance_scale
        
        sample = ablation_sampler(
            net, latents=z, 
            num_steps=args.sample_timesteps, solver="heun",
            class_labels=class_labels,
            guidance_scale=args.guidance_scale,
            eps_scaler=args.eps_scaler,
            #**model_kwargs,
        )
        
        if args.in_chans == 4:
            if (not float_equal(args.guidance_scale, 1.0)):
                sample, _ = sample.chunk(2, dim=0)  # Remove null class samples
            sample = vae.decode(sample.float() / 0.18215).sample
            
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()
        
        if args.parallel: 
            gathered_samples = [torch.zeros_like(sample) for _ in range(world_size)]
            dist.all_gather(gathered_samples, sample)
            all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

            if args.class_cond:
                gathered_labels = [torch.zeros_like(class_labels) for _ in range(world_size)]
                dist.all_gather(gathered_labels, class_labels)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        else:
            all_samples.append(sample.cpu().numpy())

            if args.class_cond:
                all_labels.append(class_labels.cpu().numpy())

        if dist_util.is_main_process() and progress_bar:
            pbar.update(sample_size * world_size)
            
    if dist_util.is_main_process() and progress_bar:
        pbar.close()
        
    model.train()  
       
    return all_samples, all_labels

def sample_and_save(args, model, sample_diffusion, device, step, save_grid=False):
    """Sample images from the model and either save them as a grid or for evaluation."""
    sampler = ddim_sampler if args.sampler == 'ddim' else heun_sampler
    
    with torch.no_grad():
        all_samples, all_labels = sampler(
            args, model, sample_diffusion, 
            num_samples=args.num_samples if not save_grid else args.sample_size, 
            sample_size=args.sample_size, 
            image_size=args.image_size, 
            num_classes=args.num_classes, 
            device=device,
            progress_bar=not save_grid,)

    if dist_util.is_main_process():
        arr = np.concatenate(all_samples, axis=0)
        arr = arr[: args.num_samples if not save_grid else args.sample_size]    

        if save_grid:
            # Save as grid image if `save_grid` is True
            torch_samples = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
            grid = make_grid(torch_samples, pad_value=0.5) 
            sample_dir = os.path.join(args.logdir, args.dataset, 'sample')
            os.makedirs(sample_dir, exist_ok=True)
            path = os.path.join(sample_dir, f'{step}.png')
            save_image(grid, path)
        else:
            # Save for evaluation purposes
            sample_dir = os.path.join(args.logdir, args.dataset, 'generate_sample')
            os.makedirs(sample_dir, exist_ok=True)
            shape_str = "x".join([str(x) for x in arr.shape[1:3]])
            out_path = os.path.join(sample_dir, f"{args.dataset}_{shape_str}_samples.npz")
            
            if args.class_cond:
                label_arr = np.concatenate(all_labels, axis=0)[: args.num_samples]
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)
            print(f"Evaluation samples saved at {out_path}")

        return arr  # Return the sampled images array for evaluation

    return None
        
def calculate_metrics(args, device, model, sample_diffusion, evaluator, ref_stats, ref_stats_spatial, ref_acts, step):
    # Sample images and get the array
    arr = sample_and_save(args, model, sample_diffusion, device, step)
    
    if dist_util.is_main_process():
        # Calculate metrics if in evaluation mode
        batches = [np.array(arr[i:i + args.sample_size]) for i in range(0, len(arr), args.sample_size)]
        sample_acts = evaluator.compute_activations(batches)
        sample_stats = evaluator.compute_statistics(sample_acts[0])
        is_score = evaluator.compute_inception_score(sample_acts[0])    
        fid = sample_stats.frechet_distance(ref_stats)
        return is_score, fid
    
    return None, None

def eval(args, device, model, ema_model, sample_diffusion, evaluator, ref_stats, ref_stats_spatial, ref_acts, eval_dir, step):
    # Evaluate net_model and ema_model
    # net_is_score, net_fid = calculate_metrics(args, model, sample_diffusion,  evaluator, ref_stats, ref_stats_spatial, ref_acts, device, step)
    # if dist_util.is_main_process():
    #     print(f"Model(NET): IS:{net_is_score:.3f}, FID:{net_fid:.3f}")
    ema_is_score, ema_fid = calculate_metrics(args, device, ema_model, sample_diffusion, evaluator, ref_stats, ref_stats_spatial, ref_acts, step)
    if dist_util.is_main_process():
        print(f"Model(EMA): IS:{ema_is_score:.3f}, FID:{ema_fid:.3f}")
        
    metrics = {
        #'IS': net_is_score,
        #'FID': net_fid,
        'IS_EMA': ema_is_score,
        'FID_EMA': ema_fid,
    }
    if dist_util.is_main_process():
        save_metrics_to_csv(args, eval_dir, metrics, step)
                
def train(args, model, ema_model, checkpoint, diffusion, sample_diffusion, train_loader, optimizer, scheduler, evaluator, ref_stats, ref_stats_spatial, ref_acts, eval_dir, device):
    model.train()
    model_size = 0
    
    for param in model.parameters():
        model_size += param.data.nelement()
        
    if dist_util.is_main_process():
        print('Model params: %.2f M' % (model_size / 1000 / 1000))
       
    # If resuming training from a checkpoint, set the start step
    start_step = checkpoint['step'] if args.resume and checkpoint else 0

    # Start training
    with trange(start_step, args.total_steps, initial=start_step, total=args.total_steps, dynamic_ncols=True, disable=not dist_util.is_main_process()) as pbar:
        trainer = Trainer(args, device, model, ema_model, optimizer, scheduler, diffusion, train_loader, start_step, pbar)
        for step in range(start_step + 1, args.total_steps + 1):
            
            loss = trainer.train_step(step)
                    
            # Sample and save images
            if args.sample_step > 0 and step % args.sample_step == 0:
                sample_and_save(args, ema_model, sample_diffusion, device, step, save_grid=True)
                    
            # Save checkpoint
            if args.save_step > 0 and step % args.save_step == 0 and step > 0:
                if dist_util.is_main_process():
                    save_checkpoint(model, optimizer, step, args, ema_model=ema_model)  
                    
            if args.parallel: 
                dist.barrier()     
            
            # Evaluate
            if args.eval and args.eval_step > 0 and step % args.eval_step == 0 and step > 0:
                eval(args, device, model, ema_model, sample_diffusion, evaluator, ref_stats, ref_stats_spatial, ref_acts, eval_dir, step)         

                              
def init(args):
    if args.parallel:
        dist_util.setup_dist()  
        local_rank = int(os.getenv('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # logger.configure(dir="logs", format_strs=['json'])
    set_random_seed(args, args.seed)
    train_loader, test_loader, input_channels, image_size = build_dataset(args)
    
    if args.eval and not args.train:
        ema_model = build_model(args).to(device)
        model = None
        
        if args.parallel:
            ema_model = torch.nn.parallel.DistributedDataParallel(ema_model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = build_model(args).to(device)
        ema_model = copy.deepcopy(model).to(device)

        if args.parallel:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
            ema_model = torch.nn.parallel.DistributedDataParallel(ema_model, device_ids=[local_rank], output_device=local_rank)

    if args.train:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=args.betas)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: warmup_cosine_lr(step, args.warmup_steps, args.total_steps, args.lr, args.final_lr, args.cosine_decay))
    else:
        optimizer = None
        scheduler = None

    if args.resume:
        if args.eval and not args.train:
            checkpoint = load_checkpoint(args.resume, ema_model=ema_model)
        else:
            checkpoint = load_checkpoint(args.resume, model=model, optimizer=optimizer, ema_model=ema_model)
        step = checkpoint['step']
    else:
        checkpoint = None
        step = 0
                
    diffusion = build_diffusion(args, use_ddim=False)
    sample_diffusion = build_diffusion(args, use_ddim=True)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config))
    
    eval_dir = os.path.join(args.logdir, args.dataset, 'evaluate')
    if dist_util.is_main_process():
        print("reading reference batch statistic...")
        ref_acts = None
        # ref_acts = evaluator.read_activations(args.fid_cache)
        ref_stats, ref_stats_spatial = evaluator.read_statistics(args.fid_cache, None)
        os.makedirs(eval_dir, exist_ok=True)
    else:
        ref_stats, ref_stats_spatial, ref_acts = None, None, None

    if args.parallel:
        dist.barrier()

    return device, train_loader, model, ema_model, checkpoint, diffusion, sample_diffusion, optimizer, scheduler, evaluator, ref_stats, ref_stats_spatial, ref_acts, eval_dir, step

            
def main():
    args = parse_args()
    device, train_loader, model, ema_model, checkpoint, diffusion, sample_diffusion, optimizer, scheduler, evaluator, ref_stats, ref_stats_spatial, ref_acts, eval_dir, step = init(args) 
    if args.train:
        train(args, model, ema_model, checkpoint, diffusion, sample_diffusion, train_loader, optimizer, scheduler, evaluator, ref_stats, ref_stats_spatial, ref_acts, eval_dir, device)
    if args.eval and not args.train:        
        assert args.resume, "Evaluation requires a checkpoint path provided with --resume"   
        eval(args, device, model, ema_model, sample_diffusion, evaluator, ref_stats, ref_stats_spatial, ref_acts, eval_dir, step)   
    if  args.parallel:  
        dist.barrier()
        dist_util.cleanup_dist()
        
if __name__ == "__main__":
    main()
