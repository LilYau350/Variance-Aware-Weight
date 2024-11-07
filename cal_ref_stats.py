import os
import glob
import numpy as np
import tensorflow.compat.v1 as tf # type: ignore
from tqdm import tqdm
from PIL import Image
from evaluations.evaluator import Evaluator  # 导入 Evaluator 类

tf.disable_v2_behavior()

data_path = './CelebA_64x64'
output_path = './fid_stats_celeba.npz'
batch_size = 32  # 设置批次大小

def preprocess_image(img_path):
    with Image.open(img_path).convert("RGB") as img:
        img = img.resize((64, 64), Image.BILINEAR)  
        img = np.array(img, dtype=np.uint8)  
        img = np.clip(img, 0, 255)  
    return img

def calculate_fid_statistics(image_paths, evaluator, batch_size):
    batches = [np.array([preprocess_image(p) for p in image_paths[i:i + batch_size]]).astype(np.uint8)
               for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches")]

    ref_acts = evaluator.compute_activations(batches)
    print(f"ref_acts type: {type(ref_acts)}")
    
    if isinstance(ref_acts, tuple):
        print(f"Number of elements in ref_acts: {len(ref_acts)}")
        for i, acts in enumerate(ref_acts):
            print(f"ref_acts[{i}] shape: {acts.shape}")
    else:
        print(f"ref_acts shape: {ref_acts.shape}")
    
    ref_stats = evaluator.compute_statistics(ref_acts[0]) 
    return ref_stats


if __name__ == "__main__":
    image_paths = glob.glob(os.path.join(data_path, '*.png'))
    print(f"Total images found: {len(image_paths)}")

    with tf.Session() as sess:
        evaluator = Evaluator(sess)

        print("Calculating FID statistics using Evaluator...")
        ref_stats = calculate_fid_statistics(image_paths, evaluator, batch_size)

    mu = ref_stats.mu
    sigma = ref_stats.sigma

    print("Saving FID statistics...")
    np.savez_compressed(output_path, mu=mu, sigma=sigma)
    print(f"FID statistics saved to {output_path}")
