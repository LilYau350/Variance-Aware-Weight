# Representation Degradation Problem in Diffusion Models

## Data Preparation

### CelebA Dataset
We follow [ScoreSDE](https://github.com/yang-song/score_sde/blob/0acb9e0ea3b8cccd935068cd9c657318fbc6ce4c/datasets.py#L121) and [FairGen](https://github.com/ermongroup/fairgen/blob/c5159789eb26699de26a4c306e6862ae3eb3cf39/src/preprocess_celeba.py#L41) for data processing.

1. **Download CelebA**: Download the dataset [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) into the `data/` directory. Select `Align&Cropped Images` and download `Img/img_align_celeba.zip`, `Anno/list_attr_celeba.txt`, and `Eval/list_eval_partition.txt`. Unzip `Img/img_align_celeba.zip` into `data/`.

2. **Preprocess CelebA**:
   ``` bash
   python ./preprocessing/preprocess_celeba.py --data_dir=/path/to/data/ --out_dir=./CelebA --partition=train
   ```
   Run this script for `--partition=[train, val, test]` to cache all necessary data. The preprocessed files will be saved in `data/`.

### ImageNet Dataset
For ImageNet, download the dataset from the [official website](https://image-net.org/download-images). We provide both online and offline preprocessing:

- **Online Processing**: Use `./datasets/data_loader.py` to perform online processing.
- **Offline Preprocessing for ImageNet-64**: Use the script at `./preprocessing/image_resizer_imagenet.py`.

We refer to the methods described in [this paper](https://arxiv.org/abs/1707.08819) and use code from [PatrykChrabaszcz's Imagenet32 Scripts](https://github.com/PatrykChrabaszcz/Imagenet32_Scripts/blob/master/image_resizer_imagent.py) and [OpenAI's guided diffusion dataset script](https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/image_datasets.py#L126). We use BOX and BICUBIC methods to ensure high-quality resizing.

### ImageNet-256
For ImageNet-256, we crop images to 256x256 and compress them using AutoencoderKL from [Diffusers](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_kl.py). We provide a preprocessing script at `./preprocessing/encode` for center cropping, random cropping, and compressing images into latents. During compression, we use a scale factor of 0.18215 to stabilize diffusion model training, and similarly, divide by 0.18215 during decompression. This follows practices from [LDM](https://github.com/CompVis/latent-diffusion) and [DiT](https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/image_datasets.py#L126).

The compressed latent codes are treated as images, except for their file extension.

## Acknowledgements

This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion). We use implementations for sampling and FID evaluation from [NVlabs/edm](https://github.com/NVlabs/edm).
