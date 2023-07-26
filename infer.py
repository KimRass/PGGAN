import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import numpy as np

from model import Generator
from celebahq import CelebAHQDataset
from utils import get_device, batched_image_to_grid, show_image

DEVICE = get_device()
gen = Generator().to(DEVICE)
gen.load_state_dict(
    torch.load("/Users/jongbeomkim/Downloads/pggan_pretrained/resol_4_iter_155200_alpha_1.pth", map_location=DEVICE)
)
gen.eval()

ROOT = "/Users/jongbeomkim/Documents/datasets/celebahq/"
ds = CelebAHQDataset(root=ROOT, split="val", resol=4)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
real_image = next(iter(dl))
grid = batched_image_to_grid(
    image=real_image, n_cols=8, mean=(0.517, 0.416, 0.363), std=(0.303, 0.275, 0.269),
)
grid = np.repeat(np.repeat(grid, repeats=1024 // resol, axis=0), repeats=1024 // resol, axis=1)
save_image(grid, path=f"""/Users/jongbeomkim/Desktop/workspace/pggan_from_scratch/generated_images/test.jpg""")

resol = 4
batch_size = 64
noise = torch.randn(batch_size, 512, 1, 1, device=DEVICE)
out = gen(noise, resol=resol, alpha=1)
grid = batched_image_to_grid(
    image=out, n_cols=8, mean=(0.517, 0.416, 0.363), std=(0.303, 0.275, 0.269),
)
grid = np.repeat(np.repeat(grid, repeats=1024 // resol, axis=0), repeats=1024 // resol, axis=1)
show_image(grid)
save_image(grid, path=f"""/Users/jongbeomkim/Desktop/workspace/pggan_from_scratch/generated_images/resol_{resol}_iter_155200.jpg""")
