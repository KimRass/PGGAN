# References:
    # https://github.com/ziwei-jiang/PGGAN-PyTorch/blob/master/train.py

import torch
from torch.optim import Adam
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from time import time

from utils import (
    get_device,
    save_parameters,
    batched_image_to_grid,
    save_image,
    resize_by_repeating_pixels,
    get_elapsed_time,
)
from model import Generator, Discriminator
from celebahq import CelebAHQDataset
from loss import get_gradient_penalty

ROOT_DIR = Path(__file__).parent
CKPT_DIR = ROOT_DIR/"pretrained"

gen = Generator()
disc = Discriminator()


def backward_hook_fn(module, grad_in, grad_out):
    # print(module)
    print(type(module))

for module in disc.modules():
    # if isinstance(module, nn.ReLU):
    # module.register_backward_hook(backward_hook_fn)


LR = 0.001
BETA1 = 0
BETA2 = 0.99
EPS = 1e-8
gen_optim = Adam(params=gen.parameters(), lr=LR, betas=(BETA1, BETA2), eps=EPS)
disc_optim = Adam(params=disc.parameters(), lr=LR, betas=(BETA1, BETA2), eps=EPS)

ROOT = "/Users/jongbeomkim/Documents/datasets/celebahq/"

batch_size = 16
alpha = 1
# resol = 4
resol = 512
ds = CelebAHQDataset(root=ROOT, split="train", resol=resol)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

start_time = time()
gen_optim.zero_grad()

real_image = next(iter(dl))
noise = torch.randn(batch_size, 512, 1, 1)

real_pred = disc(real_image, resol=resol, alpha=alpha)
fake_image = gen(noise, resol=resol, alpha=alpha)
fake_pred = disc(fake_image.detach(), resol=resol, alpha=alpha)

disc_loss = -torch.mean(real_pred) + torch.mean(fake_pred)
gp = get_gradient_penalty(
    disc=disc, resol=resol, alpha=alpha, real_image=real_image, fake_image=fake_image.detach()
)
disc_loss += LAMBDA * gp
disc_loss += EPS * torch.mean(real_pred ** 2)
disc_loss.backward()
disc_optim.step()

# noise = torch.randn(batch_size, 512, 1, 1)
# out = gen(noise, resol=resol, alpha=1)

print(get_elapsed_time(start_time))