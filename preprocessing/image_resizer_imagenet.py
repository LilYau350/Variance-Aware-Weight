import os
from PIL import Image
import numpy as np
from argparse import ArgumentParser
from multiprocessing import Pool, get_context
from tqdm import tqdm


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-i', '--in_dir', type=str, default='/data/ImageNet/ILSVRC2012', help="Input directory with original ImageNet images")
    parser.add_argument('-o', '--out_dir', type=str, default='/data/ImageNet/ImageNet-512', help="Output directory for resized images")
    parser.add_argument('-s', '--size', type=int, default=512, help="Output image size (e.g. 256 → 256×256)")
    parser.add_argument('-e', '--every_nth', type=int, default=1, help="Take every Nth class (e.g. -e 10)")
    parser.add_argument('-j', '--processes', type=int, default=8, help="Number of parallel processes")
    return parser.parse_args()


def center_crop_arr(pil_image, image_size):
    """High-quality center crop + downsample."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            (pil_image.size[0] // 2, pil_image.size[1] // 2),
            resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    new_w = round(pil_image.size[0] * scale)
    new_h = round(pil_image.size[1] * scale)
    pil_image = pil_image.resize((new_w, new_h), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2

    return arr[crop_y:crop_y + image_size, crop_x:crop_x + image_size]


def resize_img_folder(args):
    """Worker process: resize a single ImageNet class folder."""
    in_dir, out_dir, size = args

    if not os.path.exists(in_dir):
        return  # skip missing folders

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img_files = [
        f for f in os.listdir(in_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]

    for fname in img_files:
        try:
            img = Image.open(os.path.join(in_dir, fname))
            if img.mode != 'RGB':
                img = img.convert('RGB')

            cropped = center_crop_arr(img, size)
            img_resized = Image.fromarray(cropped)

            out_name = os.path.splitext(fname)[0] + ".png"
            img_resized.save(os.path.join(out_dir, out_name))

        except Exception as e:
            with open("log.txt", "a") as f:
                f.write(f"Error processing {in_dir}/{fname}: {e}\n")


if __name__ == "__main__":
    args = parse_arguments()

    in_dir = args.in_dir
    out_dir = args.out_dir
    size = args.size
    every_nth = args.every_nth
    processes = args.processes

    # Class folders come from: in_dir/train/*
    train_dir = os.path.join(in_dir, "train")

    class_folders = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])

    class_folders = class_folders[::every_nth]

    print(f"Total class folders to process: {len(class_folders)}")
    os.makedirs(out_dir, exist_ok=True)

    worker_args = []
    for cls in class_folders:
        for split in ['train', 'val']:
            in_class_dir = os.path.join(in_dir, split, cls)
            out_class_dir = os.path.join(out_dir, split, cls)

            if os.path.exists(in_class_dir):
                worker_args.append((in_class_dir, out_class_dir, size))

    ctx = get_context("spawn")
    with ctx.Pool(processes=processes) as pool:
        list(tqdm(pool.imap_unordered(resize_img_folder, worker_args),
                  total=len(worker_args)))

    print("Finished!")
