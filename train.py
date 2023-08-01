# References:
    # https://github.com/ziwei-jiang/PGGAN-PyTorch/blob/master/train.py

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from pathlib import Path
from time import time
from contextlib import nullcontext

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

# DATA_DIR = "/Users/jongbeomkim/Documents/datasets/celebahq/"
DATA_DIR = "/home/ubuntu/project/celebahq/celeba_hq"
ROOT_DIR = Path(__file__).parent
CKPT_DIR = ROOT_DIR/"pretrained"
IMG_DIR = ROOT_DIR/"generated_images"

IMG_STEPS = 1000
CKPT_STEPS = 4000

R2B = {4: 16, 8: 16, 16: 16, 32: 16, 64: 16, 128: 16, 256: 14, 512: 6, 1024: 3}
# "We start with 4×4 resolution and train the networks until we have shown the discriminator
# 800k real images in total. We then alternate between two phases: fade in the first 3-layer block
# during the next 800k images, stabilize the networks for 800k images, fade in the next 3-layer block
# during 800k images, etc."
N_IMAGES = 800_000
LAMBDA = 10
EPS = 0.001
DEVICE = get_device()
RESOLS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
N_WORKERS = 4
AUTOCAST = True

LR = 0.001
BETA1 = 0
BETA2 = 0.99
EPS = 1e-8

# "We use a minibatch size $16$ for resolutions $4^{2}$–$128^{2}$ and then gradually decrease
# the size according to $256^{2} \rightarrow 14$, $512^{2} \rightarrow 6$, $1024^{2} \rightarrow 3$
# to avoid exceeding the available memory budget."
def get_batch_size(resol):
    return R2B[resol]


def get_n_steps(batch_size):
    n_steps = N_IMAGES // batch_size
    return n_steps


def get_alpha(step, n_steps, trans_phase):
    if trans_phase:
        # "When doubling the resolution of the generator and discriminator we fade in the new layers smoothly.
        # During the transition we treat the layers that operate on the higher resolution like a residual block,
        # whose weight increases linearly from 0 to 1."
        n_steps = get_n_steps(batch_size)
        alpha = step / n_steps
    else:
        alpha = 1
    return alpha


def get_dataloader(split, batch_size, resol):
    ds = CelebAHQDataset(data_dir=DATA_DIR, split=split, resol=resol)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=N_WORKERS, pin_memory=True, drop_last=True
    )
    return dl


gen = Generator()
gen = nn.DataParallel(gen).to(DEVICE)

disc = Discriminator()
disc = nn.DataParallel(disc).to(DEVICE)

# "We train the networks using Adam with $\alpha = 0.001$, $\beta_{1} = 0$, $\beta_{2} = 0.99$,
# and $\epsilon = 10^{-8}$. We do not use any learning rate decay or rampdown, but for visualizing
# generator output at any given point during the training, we use an exponential running average
# for the weights of the generator with decay $0.999$."
gen_optim = Adam(params=gen.parameters(), lr=LR, betas=(BETA1, BETA2), eps=EPS)
disc_optim = Adam(params=disc.parameters(), lr=LR, betas=(BETA1, BETA2), eps=EPS)

gen_scaler = GradScaler()
disc_scaler = GradScaler()

ckpt_path = CKPT_DIR/"32×32_32000.pth"
gen.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

resol_idx = 3
trans_phase = False
resol = RESOLS[resol_idx]
batch_size = get_batch_size(resol)
train_dl = get_dataloader(split="train", batch_size=batch_size, resol=resol)
train_di = iter(train_dl)

n_steps = get_n_steps(batch_size)
disc_running_loss = 0
gen_running_loss = 0
step = 32000
start_time = time()
while True:
    step += 1
    alpha = get_alpha(step=step, n_steps=n_steps, trans_phase=trans_phase)

    try:
        real_image = next(train_di).to(DEVICE)
    except StopIteration:
        train_di = iter(train_di)

    # "We alternate between optimizing the generator and discriminator on a per-minibatch basis."
    ### Optimize D.
    disc_optim.zero_grad()

    # "Our latent vectors correspond to random points on a 512-dimensional hypersphere."
    noise = torch.randn(batch_size, 512, 1, 1, device=DEVICE)
    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16) if AUTOCAST else nullcontext():
        real_pred = disc(real_image, resol=resol, alpha=alpha)
        fake_image = gen(noise, resol=resol, alpha=alpha)
        fake_pred = disc(fake_image.detach(), resol=resol, alpha=alpha)

    disc_loss1 = -torch.mean(real_pred) + torch.mean(fake_pred)
    gp = get_gradient_penalty(
        disc=disc, resol=resol, alpha=alpha, real_image=real_image, fake_image=fake_image.detach()
    )
    disc_loss2 = LAMBDA * gp
    # "We use the WGAN-GP loss."
    # "We introduce a fourth term into the discriminator loss with an extremely small weight
    # to keep the discriminator output from drifting too far away from zero. We set
    # $L' = L + \epsilon_{drift}\mathbb{E}_{x \in \mathbb{P}_{r}}[D(x)^{2}]$,
    # where $\epsilon_{drift} = 0.001$."
    disc_loss3 = EPS * torch.mean(real_pred ** 2)
    disc_loss = disc_loss1 + disc_loss2 + disc_loss3

    if AUTOCAST:
        disc_scaler.scale(disc_loss).backward()
        disc_scaler.step(disc_optim)
        disc_scaler.update()
    else:
        disc_loss.backward()
        disc_optim.step()

    ### Optimize G.
    gen_optim.zero_grad()

    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
        fake_pred = disc(fake_image, resol=resol, alpha=alpha)
        gen_loss = -torch.mean(fake_pred)

    if AUTOCAST:
        gen_scaler.scale(gen_loss).backward()
        gen_scaler.step(gen_optim)
        gen_scaler.update()
    else:
        gen_loss.backward()
        gen_optim.step()

    disc_running_loss += disc_loss1.item()
    gen_running_loss += gen_loss.item()

    if (step % IMG_STEPS == 0) or (step == n_steps):
        disc_running_loss /= IMG_STEPS
        gen_running_loss /= IMG_STEPS

        print(f"""[ {resol}×{resol} ][ {step}/{n_steps} ][ {alpha:.3f} ]""", end=" ")
        print(f"""D loss: {disc_running_loss:.4f} |""", end=" ")
        print(f"""G loss: {gen_running_loss:.4f} |""", end=" ")
        print(f"""Time: {get_elapsed_time(start_time)}""")
        start_time = time()

        with torch.no_grad():
            fake_image = gen(noise, resol=resol, alpha=alpha)
            fake_image = fake_image.detach().cpu()
            grid = batched_image_to_grid(
                fake_image[: 9, ...], n_cols=3, mean=(0.517, 0.416, 0.363), std=(0.303, 0.275, 0.269)
            )
            grid = resize_by_repeating_pixels(grid, resol=resol)
            if trans_phase:
                save_path = IMG_DIR/f"""{resol // 2}×{resol // 2}to{resol}×{resol}/{step}.jpg"""
            else:
                save_path = IMG_DIR/f"""{resol}×{resol}/{step}.jpg"""
            save_image(grid, path=save_path)

        disc_running_loss = 0
        gen_running_loss = 0

    if ((resol not in [4, 8, 16]) and (step % CKPT_STEPS == 0)) or (step == n_steps):
        if trans_phase:
            save_path = CKPT_DIR/f"""{resol // 2}×{resol // 2}to{resol}×{resol}_{step}.pth"""
        else:
            save_path = CKPT_DIR/f"""{resol}×{resol}_{step}.pth"""
        save_parameters(model=gen, save_path=save_path)

    if step >= n_steps:
        if not trans_phase:
            resol_idx += 1
            resol = RESOLS[resol_idx]
            batch_size = get_batch_size(resol)
            n_steps = get_n_steps(batch_size)
            train_dl = get_dataloader(split="train", batch_size=batch_size, resol=resol)
        train_di = iter(train_dl)
        trans_phase = not trans_phase

        step = 0
