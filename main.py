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
    FlowMatching,
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
    # Enable/Disable Training and Evaluation
    parser.add_argument("--train", default=True, type=str2bool, help="Enable training")
    parser.add_argument("--eval", default=True, type=str2bool, help="Load checkpoint and evaluate FID...")

    parser.add_argument("--data_dir", type=str, default='./data', help="Path to the dataset directory")
    parser.add_argument("--dataset", type=str, default='CIFAR-10', choices=['CIFAR-10', 'Gaussian', 'CelebA', 'ImageNet', 'LSUN', 'Latent'], help="Dataset to train on")
    parser.add_argument("--patch_size", type=int, default=None, help="Patch Size for ViT, DiT, U-ViT, type is int")
    parser.add_argument("--in_chans", type=int, default=3, help="Number of input channels for the model")
    parser.add_argument("--image_size", type=int, default=32, help="Image size")
    parser.add_argument("--num_classes", type=int, default=0, help="Number of classes, type is int")
    parser.add_argument("--model", type=str, default="ADM-32", choices=model_variants, help="Model variant to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")


    # Gaussian Diffusion
    parser.add_argument("--model_mode",type=str,default="diffusion",choices=["diffusion", "flow"],
                                                    help="Choose diffusion mode: 'flow' for SDE/ODE-based modeling, 'diffusion' for DDPM-like modeling.")
    # Flow matching
    parser.add_argument("--path_type", type=str, default='linear', choices=['linear', 'cosine'], help="Path type for flow matching")    
    parser.add_argument('--sampler_type', type=str, default='sde', choices=['sde', 'ode'], help='Type of flow matching sampler to use')   
    parser.add_argument("--diffusion_term", type=str, default="sigma", choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],
                                                    help="form of diffusion coefficient in the SDE")
    parser.add_argument("--diffusion_norm", type=float, default=1.0)
    # Discrete Diffusion
    parser.add_argument("--beta_schedule", type=str, default='cosine', help="Beta schedule type 'linear', 'cosine', 'laplace', and 'power'.")
    parser.add_argument("--p", type=float, default=2, help="power for power schedule.")
    parser.add_argument("--T", type=int, default=1000, help="Number of diffusion steps")
    # loss type for diffusion or flow matching
    parser.add_argument("--mean_type", type=str, default='EPSILON', choices=['PREVIOUS_X', 'START_X', 'EPSILON', 'VELOCITY', 'VECTOR', 'SCORE'], help="Predict variable")
    parser.add_argument("--var_type", type=str, default='FIXED_LARGE', choices=['FIXED_LARGE', 'FIXED_SMALL', 'LEARNED', 'LEARNED_RANGE'], help="Variance type")
    parser.add_argument("--loss_type", type=str, default='MSE', choices=['MSE', 'RESCALED_MSE', 'KL', 'RESCALED_KL'], help="Loss type")
    parser.add_argument("--weight_type", type=str, default='constant', help="'constant', 'lambda', 'min_snr_k','vmin_snr_k', 'max_snr_k' 'debias', where k is a positive integer.")
    parser.add_argument("--gamma", type=float, default=0, help="Coefficient for loss regularization")
    parser.add_argument("--p2_gamma", type=int, default=1, help="hyperparameter for P2 weight")
    parser.add_argument("--p2_k", type=int, default=1, help="hyperparameter for P2 weight")


    # Training
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")    
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")    
    parser.add_argument("--total_steps", type=int, default=400000, help="Total training steps") 
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")        
    parser.add_argument("--class_cond", default=False, type=str2bool, help="Set class_cond to enable class-conditional generation.")
    parser.add_argument("--learn_sigma", default=False, type=str2bool, help="Set learn_sigma to enable learn distribution sigma.")    
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
    parser.add_argument("--dropout", type=float, default=0.1, help='Dropout rate of resblock')
    parser.add_argument("--cosine_decay", default=True, type=str2bool, help="Whether to use cosine learning rate decay")
    # DDP nad mixed precision training
    parser.add_argument("--parallel", default=False, type=str2bool, help="Use multi-GPU training")
    parser.add_argument('--amp', default=True, type=str2bool, help='Use AMP for mixed precision training')
    parser.add_argument('--grad_accumulation', type=int, default=1, help='Number of gradient accumulation steps (default: 1, no accumulation)')
    parser.add_argument('--resume', type=str, default=None, help='Path to the checkpoint to resume from')   


    # Logging & Sampling
    parser.add_argument("--logdir", type=str, default='./logs', help="Log directory")
    parser.add_argument("--sample_size", type=int, default=64, help="Sampling size of images")
    parser.add_argument("--sample_freq", type=int, default=10000, help="Frequency of sampling during training")        
    parser.add_argument("--sample_steps", type=int, default=18, help="Number of sample diffusion steps")   
    parser.add_argument("--class_labels", type=int, nargs="+", default=None, help="Specify the class labels used for sampling, e.g., --class_labels 207 360 387") 
    parser.add_argument("--use_classifier", type=str, default=None, help="Path to the pre-trained classifier model")
    # cfg and limited interval guidance
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='Scale factor for classifier-free guidance')       
    parser.add_argument('--t_from', type=float, default=-1, help='Starting timestep for finite interval guidance (non-negative, >= 0). Set to -1 to disable interval guidance.')
    parser.add_argument('--t_to', type=float, default=-1, help='Ending timestep for finite interval guidance (must be > t_from). Set to -1 to disable interval guidance.')    
    # which version of latnet model to choose
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    # ode/sde solver
    parser.add_argument("--solver", type=str, default='heun', choices=['ddim', 'heun', 'euler', 'dopri5'], help="Choose sampler 'ddim', 'euler' or 'heun'")
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


def build_dataset(args):
    if args.dataset == 'CIFAR-10':
        image_size = args.image_size or 32
        train_loader, test_loader = load_dataset(
            args.data_dir, args.dataset, args.batch_size, image_size, num_workers=args.num_workers, shuffle=not args.parallel)
    elif args.dataset == 'CelebA':
        image_size = args.image_size or 64
        train_loader, test_loader = load_dataset(
            args.data_dir,  args.dataset, args.batch_size, image_size, num_workers=args.num_workers, shuffle=not args.parallel)
    elif args.dataset == 'ImageNet':
        if args.image_size not in [64, 128, 256]:
            raise ValueError("Image size for ImageNet must be one of [64, 128, 256]")
        image_size = args.image_size
        train_loader, test_loader = load_dataset(
            args.data_dir, args.dataset, args.batch_size, image_size, num_workers=args.num_workers, shuffle=not args.parallel)
    elif args.dataset == 'LSUN':
        image_size = args.image_size or 256
        train_loader, test_loader = load_dataset(
            args.data_dir, args.dataset, args.batch_size, image_size, num_workers=args.num_workers, shuffle=not args.parallel)
    elif args.dataset == 'Latent':
        image_size = args.image_size or 32  # Assuming latent is 32x32x4
        train_loader, test_loader = load_dataset(
            args.data_dir, args.dataset, args.batch_size, image_size, num_workers=args.num_workers, shuffle=not args.parallel)
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
        
    return train_loader, test_loader


def build_model(args):
    unet_models = {
        "UNet-32": UNet_32,"ADM-32": ADM_32, "ADM-64": ADM_64, "ADM-128": ADM_128, 
        "ADM-256": ADM_256, "ADM-512": ADM_512, "UNet-64": UNet_64, "LDM": LDM,}
    vit_models = {
        "ViT-S": ViT_S, "ViT-B": ViT_B, "ViT-L": ViT_L, "ViT-XL": ViT_XL}
    dit_models = {
        "DiT-S": DiT_S, "DiT-B": DiT_B, "DiT-L": DiT_L, "DiT-XL": DiT_XL}
    uvit_models = {
        "U-ViT-S": UViT_S, "U-ViT-S-D": UViT_S_D, "U-ViT-M": UViT_M, 
        "U-ViT-L": UViT_L, "U-ViT-H": UViT_H}

    model_dict = {**unet_models, **vit_models, **dit_models, **uvit_models}

    if args.model not in model_dict:
        raise ValueError(f"Unsupported model variant: {args.model}")

    if any(x in args.model for x in ["UNet", "ADM", "LDM"]):
        model = model_dict[args.model](num_classes=args.num_classes, in_channels=args.in_chans, 
                                       drop_label_prob=args.drop_label_prob, dropout=args.dropout, 
                                       learn_sigma=args.learn_sigma, class_cond=args.class_cond,)
        
    elif "U-ViT" in args.model:
        model = model_dict[args.model](image_size=args.image_size, patch_size=args.patch_size,
                                       in_channels=args.in_chans, num_classes=args.num_classes) 
               
    elif "ViT" in args.model:
        model = model_dict[args.model](image_size=args.image_size, patch_size=args.patch_size,
                                       num_classes=args.num_classes, in_channels=args.in_chans,
                                       learn_sigma=args.learn_sigma, drop_rate=args.dropout, 
                                       drop_label_prob=args.drop_label_prob)

    elif "DiT" in args.model:
        model = model_dict[args.model](image_size=args.image_size, patch_size=args.patch_size,
                                       in_channels=args.in_chans, num_classes=args.num_classes,
                                       learn_sigma=args.learn_sigma,
                                       class_dropout_prob=args.drop_label_prob)
    
    return model


def build_diffusion(args, use_ddim=False):
    if args.model_mode == "diffusion":
        betas = get_named_beta_schedule(args.beta_schedule, args.T, args.p)
        timestep_respacing = (
            f"ddim{args.sample_steps}" if use_ddim and args.sample_steps < args.T else [args.T]
        )
        diffusion_kwargs = dict(
            betas=betas,
            model_mean_type=ModelMeanType[args.mean_type.upper()],
            model_var_type=ModelVarType[args.var_type.upper()],
            loss_type=LossType[args.loss_type.upper()],
            rescale_timesteps=True,
            mse_loss_weight_type=args.weight_type,
            gamma=args.gamma,
            p2_gamma=args.p2_gamma,
            p2_k=args.p2_k,
        )
        if use_ddim:
            return SpacedDiffusion(
                use_timesteps=space_timesteps(args.T, timestep_respacing),
                **diffusion_kwargs,
            )
        else:
            return GaussianDiffusion(**diffusion_kwargs)

    elif args.model_mode == "flow":
        flow_kwargs = dict(
            model_mean_type=ModelMeanType[args.mean_type.upper()],
            mse_loss_weight_type=args.weight_type,
            diffusion_term=args.diffusion_term,   
            diffusion_norm=args.diffusion_norm,         
            path_type=args.path_type,   
            sampler_type=args.sampler_type,         
            p2_gamma=args.p2_gamma,
            p2_k=args.p2_k,
            atol = args.atol,
            rtol = args.rtol,
        )
        return FlowMatching(**flow_kwargs)
    
    else:
        raise ValueError(f"Unsupported model_mode: {args.model_mode}")
               
           
def eval(args, **kwargs):
    model, ema_model, eval_dir, step = (kwargs['model'], kwargs['ema_model'], kwargs['eval_dir'], kwargs['step'])
    # Evaluate net_model and ema_model
    # net_is_score, net_fid, net_sfid, net_pre, net_rec = calculate_metrics(args, model, **kwargs)
    # if dist_util.is_main_process():
    #     print(f"Model(NET): IS:{net_is_score:.2f}, FID:{net_fid:.2f}, sFID:{net_sfid:.2f}, Pre.:{net_pre:.2f}, Rec.:{net_rec:.2f}")
    ema_is_score, ema_fid, ema_sfid, ema_pre, ema_rec = calculate_metrics(args, ema_model, **kwargs)
    if dist_util.is_main_process():
        print(f"Model(EMA): IS:{ema_is_score:.2f}, FID:{ema_fid:.2f}, sFID:{ema_sfid:.2f}, Pre:{ema_pre:.2f}, Rec:{ema_rec:.2f}")
        
    metrics = {
        # 'IS (Net)': net_is_score,
        # 'FID (Net)': net_fid,
        # 'sFID (Net)': net_sfid,        
        # 'Precision (Net)': net_pre,
        # 'Recall (Net)': net_rec,
        'IS (EMA)': ema_is_score,
        'FID (EMA)': ema_fid,
        'sFID (EMA)': ema_sfid,        
        'Pre. (EMA)': ema_pre,
        'Rec. (EMA)': ema_rec,
    }
    if dist_util.is_main_process():
        save_metrics_to_csv(args, eval_dir, metrics, step)
                
def train(args, **kwargs):
    
    model, ema_model, checkpoint, diffusion, sample_diffusion, train_loader, optimizer, scheduler, device = (
        kwargs['model'], kwargs['ema_model'], kwargs['checkpoint'], 
        kwargs['diffusion'], kwargs['sample_diffusion'], kwargs['train_loader'],
        kwargs['optimizer'], kwargs['scheduler'], kwargs['device']
    )
    
    # model.train()
    model_size = sum(param.data.nelement() for param in model.parameters())
        
    if dist_util.is_main_process():
        print('Model params: %.2f M' % (model_size / 1_000_000))
        print('Total batch size (per update step): %d' % (args.batch_size * args.grad_accumulation))

    # If resuming training from a checkpoint, set the start step
    start_step = checkpoint['step'] if args.resume and checkpoint else 0

    # Start training
    with trange(start_step, args.total_steps, initial=start_step, total=args.total_steps, 
                dynamic_ncols=True, disable=not dist_util.is_main_process()) as pbar:
        trainer = Trainer(args, device, model, ema_model, optimizer, scheduler, diffusion, train_loader, start_step, pbar)
        for step in range(start_step + 1, args.total_steps + 1):
            
            loss = trainer.train_step(step)      
            # Sample and save images
            if args.sample_freq > 0 and step % args.sample_freq == 0:
                # sample_and_save(args, step, device, ema_model, sample_diffusion, save_grid=True)
                generate_samples(args, step, device, ema_model, sample_diffusion, save_grid=True)
    
            # Save checkpoint
            if args.save_step > 0 and step % args.save_step == 0 and step > 0:
                if dist_util.is_main_process():
                    save_checkpoint(args, step, model, optimizer, ema_model=ema_model)  
        
            # Evaluate
            if args.eval and args.eval_step > 0 and step % args.eval_step == 0 and step > 0:
                eval(args, **{**kwargs, 'step': step})  
                
            if args.parallel: 
                dist.barrier()                        


def init(args):
    if args.parallel:
        dist_util.setup_dist()  
        local_rank = int(os.getenv('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    set_random_seed(args, args.seed)
    train_loader, _ = build_dataset(args)
    
    diffusion = build_diffusion(args, use_ddim=False)
    sample_diffusion = build_diffusion(args, use_ddim=True)

    
    if args.eval and not args.train:
        ema_model = build_model(args).to(device)
        model = None
        
        if args.parallel:
            ema_model = DDP(ema_model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = build_model(args).to(device)
        ema_model = copy.deepcopy(model).to(device)

        if args.parallel:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            ema_model = DDP(ema_model, device_ids=[local_rank], output_device=local_rank)

    if args.train:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay, eps=args.eps)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda(args))
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

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config))

    eval_dir = os.path.join(args.logdir, args.dataset, 'evaluate')
    if dist_util.is_main_process():
        print("warming up TensorFlow...")
        # This will cause TF to print a bunch of verbose stuff now rather
        # than after the next print(), to help prevent confusion.
        evaluator.warmup()

        print("computing reference batch activations...")
        ref_acts = evaluator.read_activations(args.ref_batch)
        print("computing/reading reference batch statistics...")
        ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref_batch, ref_acts)
        
        os.makedirs(eval_dir, exist_ok=True)
    else:
        ref_acts, ref_stats, ref_stats_spatial = None, None, None

    if args.parallel:
        dist.barrier()

    return {
        'device': device, 'train_loader': train_loader, 'model': model, 'ema_model': ema_model, 'checkpoint': checkpoint,
        'diffusion': diffusion, 'sample_diffusion': sample_diffusion, 'optimizer': optimizer, 'scheduler': scheduler,'evaluator': evaluator, 
        'ref_acts': ref_acts, 'ref_stats': ref_stats, 'ref_stats_spatial': ref_stats_spatial, 'eval_dir': eval_dir, 'step': step }

def main():
    args = parse_args()
    init_params = init(args)  
    if args.train:
        train(args, **init_params)  
    if args.eval and not args.train:        
        assert args.resume, "Evaluation requires a checkpoint path provided with --resume"   
        eval(args, **init_params)  
    if args.parallel:  
        dist.barrier()
        dist_util.cleanup_dist()
        
if __name__ == "__main__":
    main()                                                                                                          
