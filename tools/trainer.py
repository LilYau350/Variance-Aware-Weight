import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tools import dist_util, logger
from .resample import LossAwareSampler, UniformSampler, create_named_schedule_sampler

def ema(source, target, decay):
    with torch.no_grad():
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * decay + source_dict[key].data * (1 - decay))

class Trainer:
    def __init__(self, args, device, model, ema_model, optimizer, scheduler, diffusion, train_loader, start_step, pbar=None):
        self.args = args
        self.device = device        
        self.model = model
        self.ema_model = ema_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.datalooper = iter(train_loader)
        self.schedule_sampler = create_named_schedule_sampler(args.sampler_type, diffusion)
        self.scaler = GradScaler() if args.amp else None
        self.start_step = start_step        
        self.pbar = pbar

    def _get_next_batch(self):
        try:
            images, labels = next(self.datalooper)
        except StopIteration:
            self.datalooper = iter(self.train_loader)
            images, labels = next(self.datalooper)
        return images.to(self.device), labels.to(self.device) if self.args.class_cond else None

    def _compute_loss(self, images, labels):
        model_kwargs = {"y": labels} if self.args.class_cond else {}
        t, weights = self.schedule_sampler.sample(images.shape[0], device=self.device)
        loss_dict = self.diffusion.training_losses(self.model, images, t, model_kwargs=model_kwargs)
        
        # Update sampler with local losses if using LossAwareSampler
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(t, loss_dict["loss"].detach())
        
        return (loss_dict["loss"] * weights).mean()

    def _apply_gradient_clipping(self):
        if self.args.grad_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

    def _update_ema(self):
        if dist_util.is_main_process():
            ema(self.model, self.ema_model, self.args.ema_decay)

    def train_step(self, step):
        if self.args.parallel:
            self.train_loader.sampler.set_epoch(step)
        
        images, labels = self._get_next_batch()
        self.optimizer.zero_grad()
        
        # Mixed precision training
        if self.args.amp:
            with autocast():
                loss = self._compute_loss(images, labels)
            self.scaler.scale(loss).backward()
            if self.args.grad_clip:
                self.scaler.unscale_(self.optimizer)
                self._apply_gradient_clipping()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self._compute_loss(images, labels)
            loss.backward()
            self._apply_gradient_clipping()
            self.optimizer.step()
        
        # Update scheduler
        self.scheduler.step()
        
        if dist_util.is_main_process():
            self._update_ema()
            self.pbar.update(1)
            self.pbar.set_postfix(loss=loss.item())
        
        return loss.item()  # Return loss value for logging if needed


