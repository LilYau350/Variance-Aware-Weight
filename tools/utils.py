import argparse
import csv
import os
import re
import math
import json
import yaml 
import random
import torch
import numpy as np
from datetime import datetime
import torch.distributed as dist
from torchvision.utils import make_grid, save_image
from tools import dist_util
from tools.sampler import Sampler, Classifier
from tools.trainer import sample_from_latent
from tqdm import tqdm
from contextlib import nullcontext

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def generate_logdir(args):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(args.logdir, timestamp)
    args.logdir = logdir
     if dist_util.is_main_process():           
        os.makedirs(logdir, exist_ok=True)
        config_path = os.path.join(logdir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(vars(args), f, sort_keys=False)
        
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
    
    
def get_lr_lambda(args):
    return lambda step: warmup_cosine_lr(
        step, args.warmup_steps, args.total_steps,
        args.lr, args.final_lr, args.cosine_decay)


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
        
        
def save_checkpoint(args, step, model, optimizer, ema_model=None):
    model_name = args.model
    if "DiT" in model_name and args.model_mode == 'flow':
        model_name = 'SiT' + model_name[3:]
    if dist_util.is_main_process():
        checkpoint_dir = os.path.join(args.logdir, 'checkpoint',)
        os.makedirs(checkpoint_dir, exist_ok=True)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step
        }
        if ema_model is not None:
            state['ema_model'] = ema_model.state_dict()
        filename = f"{model_name}_{args.mean_type}_{args.path_type}_{step}.pth"
        filename = os.path.join(checkpoint_dir, filename)
        torch.save(state, filename)
        print(f"Checkpoint saved: {filename}")

def load_checkpoint(ckpt_path, model=None, optimizer=None, ema_model=None):
    if dist_util.is_main_process():
        print('==> Resuming from checkpoint..')
    assert os.path.exists(ckpt_path), 'Error: checkpoint {} not found'.format(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if model:
        model.load_state_dict(checkpoint['model'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if ema_model and 'ema_model' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_model'])
    return checkpoint


def generate_samples(args, step, device, eval_model, sample_diffusion, save_grid=False):
    """Sample images from the model and either save them as a grid or for evaluation."""
    classifier = Classifier(args, device, eval_model) if args.use_classifier else None
    sampler = Sampler(args, device, eval_model, sample_diffusion, classifier=classifier)
    
    with torch.no_grad():
        all_samples, all_labels = sampler.sample(
            num_samples=args.num_samples if not save_grid else 64, 
            sample_size=args.sample_size, 
            image_size=args.image_size, 
            num_classes=args.num_classes, 
            progress_bar=not save_grid,)
        
    return save_images(args, step, all_samples,all_labels, save_grid)    
    
def save_images(args, step, samples, labels, save_grid=False):
    """Save sampled images as a grid."""
    if dist_util.is_main_process():
        arr = np.concatenate(samples, axis=0)
        arr = arr[: args.num_samples if not save_grid else 64]    
        
        if save_grid:
            # Save as grid image if 'save_grid' is True
            torch_samples = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
            grid = make_grid(torch_samples, pad_value=0.5)
            sample_dir = os.path.join(args.logdir, 'sample')
            os.makedirs(sample_dir, exist_ok=True)
            path = os.path.join(sample_dir, f'{step}.png')
            save_image(grid, path)
        else:
            # Save for evaluation purposes
            sample_dir = os.path.join(args.logdir, 'generate_sample')
            os.makedirs(sample_dir, exist_ok=True)
            shape_str = "x".join([str(x) for x in arr.shape[1:3]])
            out_path = os.path.join(sample_dir, f"{args.dataset}_{shape_str}_samples.npz")
            
            if args.class_cond:
                label_arr = np.concatenate(labels, axis=0)[: args.num_samples]
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)
            print(f"Evaluation samples saved at {out_path}")

        return arr  # Return the sampled images array for evaluation
        
    return None  

def calculate_metrics(args, eval_model, **kwargs):
    
    step, device, sample_diffusion, evaluator, ref_acts, ref_stats, ref_stats_spatial = (
        kwargs['step'], kwargs['device'],  kwargs['sample_diffusion'], kwargs['evaluator'], 
        kwargs['ref_acts'], kwargs['ref_stats'], kwargs['ref_stats_spatial'])
    
    # Sample images and get the array
    arr = generate_samples(args, step, device, eval_model, sample_diffusion)
    if dist_util.is_main_process():
        # Calculate metrics if in evaluation mode
        sample_batch = [np.array(arr[i:i + args.sample_size]) for i in range(0, len(arr), args.sample_size)]
        sample_acts = evaluator.compute_activations(sample_batch)

        sample_stats, sample_stats_spatial = tuple(evaluator.compute_statistics(x) for x in sample_acts)
        is_score = evaluator.compute_inception_score(sample_acts[0])   
        fid = sample_stats.frechet_distance(ref_stats)
        sfid = sample_stats_spatial.frechet_distance(ref_stats_spatial)
        pre, rec = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
        return is_score, fid, sfid, pre, rec
    
    return None, None, None, None, None
    
@torch.no_grad()
def _topk_correct(logits, target, ks=(1,5)):
    num_classes = logits.size(1)
    assert all(0 < k <= num_classes for k in ks), f"topk={ks} exceeds num_classes={num_classes}"
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # [B, maxk]
    pred = pred.t()                                                # [maxk, B]
    correct = pred.eq(target.view(1, -1).expand_as(pred))          # [maxk, B] bool
    return [correct[:k].reshape(-1).float().sum() for k in ks]

@torch.no_grad()
def eval_accuracy(args, val_loader, model, device, desc="ValDataset", topk=(1,5)):
    model.eval()
    total = torch.zeros(1, device=device)
    correct = {k: torch.zeros(1, device=device) for k in topk}

    pbar = tqdm(total=len(val_loader), desc=desc,
                disable=not dist_util.is_main_process(),
                dynamic_ncols=True, leave=False)

    for images, labels in val_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if args.in_chans == 4:
            images = sample_from_latent(images, args.latent_scale) 
             
        with torch.cuda.amp.autocast(enabled=args.amp):
            t = torch.zeros(images.shape[0], dtype=torch.long, device=images.device) 
            _, logits = model(images, t, labels)

        corr_locals = _topk_correct(logits, labels, ks=topk)
        n_local = torch.tensor(labels.size(0), device=device, dtype=torch.float32)

        if dist.is_available() and dist.is_initialized():
            corr_vec = torch.stack(corr_locals + [n_local], dim=0)
            dist.all_reduce(corr_vec, op=dist.ReduceOp.SUM)
            *corr_locals, n_local = corr_vec.unbind(0)

        total += n_local
        for k, c in zip(topk, corr_locals):
            correct[k] += c

        # if dist_util.is_main_process():
        denom = max(total.item(), 1.0)
        top1 = (correct.get(1, torch.tensor(0., device=device)).item() / denom) * 100
        top5 = (correct.get(5, torch.tensor(0., device=device)).item() / denom) * 100
        pbar.set_postfix_str(f"Top(1,5)=({top1:.2f},{top5:.2f})%")
        pbar.update(1)

    if dist_util.is_main_process():
        pbar.close()
        
    return top1, top5

def save_metrics_to_csv(args, metrics, step):
    csv_filename = os.path.join(args.logdir, f"metrics.csv")
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Step'] + list(metrics.keys()))
        # writer.writerow([step] + list(metrics.values()))
        formatted_values = [f"{value:.2f}" if isinstance(value, (float, int)) else value for value in metrics.values()]
        writer.writerow([step] + formatted_values)

