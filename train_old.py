# References:
    # https://www.kaggle.com/code/heonh0/pggan-progressive-growing-gan-pggan-pytorch
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

R2B = {4: 16, 8: 16, 16: 16, 32: 16, 64: 16, 128: 16, 256: 14, 512: 6, 1024: 3}
# "We start with $4 \times 4$ resolution and train the networks until we have shown the discriminator
# 800k real images in total. We then alternate between two phases: fade in the first 3-layer block
# during the next 800k images, stabilize the networks for 800k images, fade in the next 3-layer block
# during 800k images, etc."
N_IMAGES = 800_000
# "When doubling the resolution of the generator and discriminator we fade in the new layers smoothly.
# During the transition we treat the layers that operate on the higher resolution like a residual block,
# whose weight increases linearly from 0 to 1."
ALPHAS = np.linspace(0, 1, N_IMAGES)

# "We use a minibatch size $16$ for resolutions $4^{2}$–$128^{2}$ and then gradually decrease
# the size according to $256^{2} \rightarrow 14$, $512^{2} \rightarrow 6$, $1024^{2} \rightarrow 3$
# to avoid exceeding the available memory budget."
def get_batch_size(resol):
    return R2B[resol]


def get_n_iters(batch_size):
    n_iters = N_IMAGES // batch_size
    return n_iters


def get_alpha(iter_):
    return ALPHAS[iter_ - 1]


DEVICE = get_device()
gen = Generator().to(DEVICE)
disc = Discriminator().to(DEVICE)

# "We train the networks using Adam with $\alpha = 0.001$, $\beta_{1} = 0$, $\beta_{2} = 0.99$,
# and $\epsilon = 10^{-8}$. We do not use any learning rate decay or rampdown, but for visualizing
# generator output at any given point during the training, we use an exponential running average
# for the weights of the generator with decay $0.999$."
LR = 0.001
BETA1 = 0
BETA2 = 0.99
EPS = 1e-8
gen_optim = Adam(params=gen.parameters(), lr=LR, betas=(BETA1, BETA2), eps=EPS)
disc_optim = Adam(params=disc.parameters(), lr=LR, betas=(BETA1, BETA2), eps=EPS)

gen_scaler = GradScaler()
disc_scaler = GradScaler()

RESOLS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
res_idx = 2
resol = RESOLS[res_idx]
ROOT = "/home/ubuntu/project/celebahq/celeba_hq"
# ROOT = "/Users/jongbeomkim/Documents/datasets/celebahq/"
ds = CelebAHQDataset(root=ROOT, split="train", resol=resol)
TRANS_PHASE = False
batch_size = get_batch_size(resol)
N_WORKERS = 4
# N_WORKERS = 0
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=N_WORKERS, pin_memory=True, drop_last=True)
LAMBDA = 10
EPS = 0.001

ckpt_path = CKPT_DIR/"16x16/resol_16_iter_240000.pth"
gen.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
_, _, _, iter_ = ckpt_path.stem.split("_")
# resol = int(resol)
iter_ = int(iter_)
# res_idx = 0

# iter_ = 224_000
breaker = False
start_time = time()
while True:
    if breaker:
        break

    for batch, real_image in enumerate(dl, start=1):
        iter_ += 1

        if TRANS_PHASE:
            alpha = get_alpha(iter_)
        else:
            alpha = 1

        real_image = real_image.to(DEVICE)
        disc_optim.zero_grad()

        noise = torch.randn(batch_size, 512, 1, 1, device=DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
            real_pred = disc(real_image, resol=resol, alpha=alpha)
            fake_image = gen(noise, resol=resol, alpha=alpha)
            fake_pred = disc(fake_image.detach(), resol=resol, alpha=alpha)

        disc_loss = -torch.mean(real_pred) + torch.mean(fake_pred)
        gp = get_gradient_penalty(
            disc=disc, resol=resol, alpha=alpha, real_image=real_image, fake_image=fake_image.detach()
        )
        disc_loss += LAMBDA * gp
        disc_loss += EPS * torch.mean(real_pred ** 2)

        disc_scaler.scale(disc_loss).backward()
        disc_scaler.step(disc_optim)
        disc_scaler.update()

        gen_optim.zero_grad()

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
            fake_pred = disc(fake_image, resol=resol, alpha=alpha)
            gen_loss = -torch.mean(fake_pred)

        gen_scaler.scale(gen_loss).backward()
        gen_scaler.step(gen_optim)
        gen_scaler.update()

        if iter_ % (N_IMAGES // 100) == 0:
            print(f"""[ {resol} ][ {iter_}/{N_IMAGES} ][ {alpha: .3f} ]""", end=" ")
            print(f"""G loss: {gen_loss.item(): .6f} | D loss: {disc_loss.item(): .6f}""", end=" ")
            print(f""" | Time: {get_elapsed_time(start_time)}""")
            start_time = time()

        if iter_ % (N_IMAGES // 50) == 0:
            fake_image = fake_image.detach().cpu()
            grid = batched_image_to_grid(
                fake_image[: 3, ...], n_cols=3, mean=(0.517, 0.416, 0.363), std=(0.303, 0.275, 0.269)
            )
            grid = resize_by_repeating_pixels(grid, resol=resol)
            phase = "transition_phase_" if TRANS_PHASE else ""
            save_image(
                grid, path=ROOT_DIR/f"""generated_images/{phase}resol_{resol}_iter_{iter_}.jpg"""
            )

            save_parameters(
                model=gen,
                save_path=CKPT_DIR/f"""resol_{resol}_iter_{iter_}{phase}.pth"""
            )

        if iter_ >= N_IMAGES:
            if TRANS_PHASE:
                TRANS_PHASE = False
                iter_ = 0
            else:
                if resol == RESOLS[-1]:
                    breaker = True
                    break
                res_idx += 1
                resol = RESOLS[res_idx]
                batch_size = get_batch_size(resol)
                TRANS_PHASE = True
                iter_ = 0
