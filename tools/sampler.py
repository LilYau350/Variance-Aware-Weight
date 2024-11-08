import torch
from tqdm import tqdm
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from tools import dist_util
from cfg_edm import ablation_sampler, float_equal, Net
from models.unet import EncoderUNetModel

class Sampler:
    def __init__(self, args, device, model, ema_model, diffusion):
        self.args = args
        self.device = device        
        self.model = model
        self.ema_model = ema_model        
        self.diffusion = diffusion
        self.classifier = self.load_classifier() if args.use_classifier else None


    def create_classifier(self):
        attention_ds = [self.args.image_size // res for res in self.model.attention_resolutions]
        classifier_kwargs = {
            'image_size': self.model.image_size,
            'in_channels': self.args.in_chans,
            'model_channels': self.model.model_channels,
            'out_channels': self.model.num_classes,
            'num_res_blocks': self.model.num_res_blocks,
            'attention_resolutions': tuple(attention_ds),
            'channel_mult': self.model.channel_mult,
            'num_head_channels': self.model.num_head_channels,
            'use_scale_shift_norm': self.model.use_scale_shift_norm,
            'resblock_updown': self.model.resblock_updown,
            'pool': "attention",
        }
        return EncoderUNetModel(**classifier_kwargs)

    def load_classifier(self):
        classifier = self.create_classifier()
        classifier.load_state_dict(torch.load(self.args.use_classifier, map_location="cpu"))
        classifier.to(self.device)
        classifier.eval()
        return classifier

    def sync_ema_model(self):
        """Synchronize EMA model parameters across distributed devices."""
        for param in self.ema_model.parameters():
            dist.broadcast(param.data, src=0)

    def cond_fn(self, x, t, y=None, classifier_scale=1.0):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = self.classifier(x_in, t)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(log_probs)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale

    def model_fn(self, x, t, y=None):
        return self.model(x, t, y if self.args.class_cond else None)

    def ddim_sampler(self, num_samples, sample_size, image_size, num_classes, progress_bar=False):
        self.model.eval()
        all_samples, all_labels = [], []
        world_size = dist.get_world_size() if self.args.parallel else 1
        
        if self.args.parallel:
            self.sync_ema_model()
            dist.barrier()

        if progress_bar and dist_util.is_main_process():
            pbar = tqdm(total=num_samples, desc="Generating Samples (DDIM)")

        while len(all_samples) * sample_size < num_samples:
            classes = torch.randint(low=0, high=num_classes, size=(sample_size,), device=self.device) if self.args.class_cond else None
            sample = self.diffusion.ddim_sample_loop(
                self.model if not self.classifier else self.model_fn,
                (sample_size, 3, image_size, image_size),
                device=self.device,
                model_kwargs={"y": classes} if self.args.class_cond else {},
                cond_fn=(lambda x, t, y: self.cond_fn(x, t, y, self.args.guidance_scale)) if self.classifier else None,
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            sample = sample.permute(0, 2, 3, 1).contiguous()

            if self.args.parallel:
                gathered_samples = [torch.zeros_like(sample) for _ in range(world_size)]
                dist.all_gather(gathered_samples, sample)
                all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

                if self.args.class_cond:
                    gathered_labels = [torch.zeros_like(classes) for _ in range(world_size)]
                    dist.all_gather(gathered_labels, classes)
                    all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            else:
                all_samples.append(sample.cpu().numpy())
                if self.args.class_cond:
                    all_labels.append(classes.cpu().numpy())

            if dist_util.is_main_process() and progress_bar:
                pbar.update(sample_size * world_size)

        return all_samples, all_labels

    def heun_sampler(self, num_samples, sample_size, image_size, num_classes, progress_bar=False):
        self.model.eval()
        all_samples, all_labels = [], []
        world_size = dist.get_world_size() if self.args.parallel else 1
        
        if self.args.parallel:
            self.sync_ema_model()
            dist.barrier()

        if progress_bar and dist_util.is_main_process():
            pbar = tqdm(total=num_samples, desc="Generating Samples (Heun)")

        if self.args.in_chans == 4:
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(self.device)

        net = Net(model=self.model, img_channels=self.args.in_chans, img_resolution=image_size,
                  noise_schedule=self.args.beta_schedule, amp=self.args.amp).to(self.device)

        while len(all_samples) * sample_size < num_samples:
            y_cond = torch.randint(low=0, high=num_classes, size=(sample_size,), device=self.device) if self.args.class_cond else None
            z = torch.randn([sample_size, net.img_channels, net.img_resolution, net.img_resolution], device=self.device)
            if not float_equal(self.args.guidance_scale, 1.0):
                z = torch.cat((z, z), dim=0)
                y_uncond = torch.randint(low=num_classes, high=num_classes + 1, size=(sample_size,), device=self.device)
                class_labels = torch.cat((y_cond, y_uncond), dim=0) if y_cond is not None else None
            else:
                class_labels = y_cond

            sample = ablation_sampler(net, latents=z, num_steps=self.args.sample_timesteps, solver="heun",
                                      class_labels=class_labels, guidance_scale=self.args.guidance_scale,
                                      eps_scaler=self.args.eps_scaler)
            
            if self.args.in_chans == 4:
                if not float_equal(self.args.guidance_scale, 1.0):
                    sample, _ = sample.chunk(2, dim=0)
                sample = vae.decode(sample.float() / 0.18215).sample

            sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            sample = sample.permute(0, 2, 3, 1).contiguous()

            if self.args.parallel:
                gathered_samples = [torch.zeros_like(sample) for _ in range(world_size)]
                dist.all_gather(gathered_samples, sample)
                all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

                if self.args.class_cond:
                    gathered_labels = [torch.zeros_like(class_labels) for _ in range(world_size)]
                    dist.all_gather(gathered_labels, class_labels)
                    all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            else:
                all_samples.append(sample.cpu().numpy())
                if self.args.class_cond:
                    all_labels.append(class_labels.cpu().numpy())

            if dist_util.is_main_process() and progress_bar:
                pbar.update(sample_size * world_size)
        
        return all_samples, all_labels

    def sample(self, num_samples, sample_size, image_size, num_classes, method="ddim", progress_bar=False):
        if method == "ddim":
            return self.ddim_sampler(num_samples, sample_size, image_size, num_classes, progress_bar)
        elif method == "heun":
            return self.heun_sampler(num_samples, sample_size, image_size, num_classes, progress_bar)
