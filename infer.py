import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Generator
from celebahq import CelebAHQDataset
from torch_utils import get_device, batched_image_to_grid, show_image

DEVICE = get_device()
gen = Generator().to(DEVICE)
gen.load_state_dict(
    torch.load("/Users/jongbeomkim/Downloads/pggan_parameters/resol_4_iter_446400.pth", map_location=DEVICE)
)
gen.eval()

ROOT = "/Users/jongbeomkim/Documents/datasets/celebahq/"
ds = CelebAHQDataset(root=ROOT, split="val", resol=4)
batch_size = 64
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
# for batch, real_image in enumerate(dl, start=1):
#     batch
real_image = next(iter(dl))

noise = torch.randn(batch_size, 512, 1, 1, device=DEVICE)
out = gen(noise, resol=4, alpha=1)
grid = batched_image_to_grid(
    image=out, n_cols=8, mean=(0.517, 0.416, 0.363), std=(0.303, 0.275, 0.269),
)
show_image(grid)
