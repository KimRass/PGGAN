import torch
from pathlib import Path
from time import time
from tqdm.auto import tqdm
import argparse

import config
from utils import (
    image_to_grid,
    save_image,
)
from model import Generator


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_size", type=int, required=True)
    parser.add_argument("--n_images", type=int, required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    gen = Generator()
    if config.N_GPUS > 0:
        DEVICE = torch.device("cuda")
        gen = gen.to(DEVICE)

    ckpt = torch.load(config.PRETRAINED, map_location=DEVICE)
    if config.N_GPUS > 1 and config.MULTI_GPU:
        gen.module.load_state_dict(ckpt["G"])
    else:
        gen.load_state_dict(ckpt["G"])

    ### Generate images
    gen.eval()
    with torch.no_grad():
        for idx in tqdm(range(args.n_images)):
            noise = torch.randn(1, 512, 1, 1, device=DEVICE)
            fake_image = gen(noise, img_size=args.img_size, alpha=1)

            fake_image = fake_image.detach().cpu()
            grid = image_to_grid(fake_image, n_cols=1, value_range=(-1, 1))
            save_path = Path(__file__).parent/\
                f"""generated_images/{args.img_size}×{args.img_size}/{idx}.jpg"""
            save_image(grid, path=save_path)
