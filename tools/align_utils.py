# align_utils.py
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tools.encoders import load_encoders
import torchvision.transforms as transforms
from torch.cuda.amp import autocast

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

# Initialize encoders
def initialize_encoders(args, device):
    encoder, _, _ = load_encoders(args.enc_type, device, args.image_size * 8)
    encoder = encoder[0] if isinstance(encoder, list) else encoder
    encoder.eval()
    return encoder

# Preprocess image based on encoder type
def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = transforms.Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x

# Get features from encoder
def get_feature(args, images, encoder):
    images = preprocess_raw_image(images, args.enc_type)
    with torch.no_grad():
        with autocast():
            features = encoder.forward_features(images)
            if 'mocov3' in args.enc_type: features = features[:, 1:]
            if 'dinov2' in args.enc_type: features = features['x_norm_patchtokens']
    return features
    
