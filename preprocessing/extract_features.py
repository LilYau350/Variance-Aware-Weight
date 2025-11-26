import argparse
import h5py
import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import numpy as np
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from tools.encoders import load_encoders
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from torch.cuda.amp import autocast

'''
ImageNet_Feature.h5
├── train_latents  # Shape: (num_train_samples, latent_dim)
├── train_features  # Shape: (num_train_samples, feature_dim)
├── train_labels   # Shape: (num_train_samples,)
├── val_latents    # Shape: (num_val_samples, latent_dim)
├── val_features  # Shape: (num_val_samples, feature_dim)
└── val_labels     # Shape: (num_val_samples,)
'''



CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing functions
def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

# Initialize VAE model
def initialize_vae(args, device):
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    return vae

# Initialize encoders
def initialize_encoders(args, device, resolution=256):
    encoder, _, _ = load_encoders(args.enc_type, device, resolution)
    encoder = encoder[0] if isinstance(encoder, list) else encoder
    encoder.eval()
    return encoder

# Preprocess image based on encoder type
def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = (x + 1) / 2
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = transforms.Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = (x + 1) / 2
        x = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = (x + 1) / 2
        x = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = (x + 1) / 2
        x = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    return x

# Load ImageNet dataset
def load_imagenet(input, image_size, batch_size):
    transform = transforms.Compose([ 
        transforms.Lambda(lambda img: center_crop_arr(img, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = datasets.ImageFolder(root=f"{input}/train", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{input}/val", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader

# Get latent representation from VAE
def get_latent(images, device, vae):
    images = images.to(device)
    with torch.no_grad():
        with autocast():
            latent_dist = vae.encode(images).latent_dist
            latents = torch.cat([latent_dist.mean, latent_dist.std], dim=1)
    return latents

# Get features from encoder
def get_feature(args, images, device, encoder):
    images = images.to(device)
    images = preprocess_raw_image(images, args.enc_type)
    with torch.no_grad():
        with autocast():
            features = encoder.forward_features(images)
            if 'mocov3' in args.enc_type: features = features[:, 1:]
            if 'dinov2' in args.enc_type: features = features['x_norm_patchtokens']
    return features

# Save dataset to .h5 format
def save_dataset_to_h5(args, data_loader, f, dataset_name, device, vae, encoder):
    latents_dataset = None  
    features_dataset = None
    labels_dataset = None
    
    for batch_idx, (images, labels) in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Compressing {dataset_name}"):
        latents = get_latent(images, device, vae)
        features = get_feature(args, images, device, encoder)
        
        if latents_dataset is None:
            num_latents = len(data_loader.dataset)
            latents_shape = latents.shape[1:]  # e.g., (D,)
            features_shape = features.shape[1:]  # e.g., (D,)
            
            latents_dataset = f.create_dataset(
                f'{dataset_name}_latents', (num_latents, *latents_shape), dtype='float32'
            )

            features_dataset = f.create_dataset(
                f'{dataset_name}_features', (num_latents, *features_shape), dtype='float32'
            )
            
            labels_dataset = f.create_dataset(
                f'{dataset_name}_labels', (num_latents,), dtype='int64'  
            )
        
        start_idx = batch_idx * data_loader.batch_size
        end_idx = start_idx + latents.size(0)
        
        latents_dataset[start_idx:end_idx] = latents.cpu().numpy()
        features_dataset[start_idx:end_idx] = features.cpu().numpy()
        labels_dataset[start_idx:end_idx] = labels.cpu().numpy()
        f.flush()
    
    
# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Extract features from raw images")
    parser.add_argument("--input", type=str, default='/data/ImageNet/ILSVRC2012', help="Path to input image folder")
    parser.add_argument("--output", type=str, default='/data/ImageNet', help="Output file for features")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for processing images")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--enc_type", type=str, default="dinov2-vit-b", help="Encoder specification")
    parser.add_argument("--image_size", type=int, default=256, help="Size of input images")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Initialize the VAE model and encoder
    vae = initialize_vae(args, device)
    encoder = initialize_encoders(args, device, args.image_size)

    # Load datasets
    train_loader, val_loader = load_imagenet(args.input, args.image_size, args.batch_size)

    # Output path for h5 file
    h5_file = os.path.join(args.output, "ImageNet_Feature.h5")

    # Save datasets to h5 file
    with h5py.File(h5_file, 'w') as f:
        save_dataset_to_h5(args, train_loader, f, "train", device, vae, encoder)
        save_dataset_to_h5(args, val_loader, f, "val", device, vae, encoder)