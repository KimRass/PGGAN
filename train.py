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
import config

from utils import (
    get_device,
    save_checkpoint,
    image_to_grid,
    save_image,
    get_elapsed_time,
)
from model import Generator, Discriminator
from celebahq import CelebAHQDataset
from loss import get_gradient_penalty

RESOLS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

ROOT_DIR = Path(__file__).parent
CKPT_DIR = ROOT_DIR/"checkpoints"
IMG_DIR = ROOT_DIR/"generated_images"

DEVICE = get_device()

# "We use a minibatch size $16$ for resolutions $4^{2}$–$128^{2}$ and then gradually decrease
# the size according to $256^{2} \rightarrow 14$, $512^{2} \rightarrow 6$, $1024^{2} \rightarrow 3$
# to avoid exceeding the available memory budget."
def get_batch_size(resol):
    return config.RESOL_BATCH_SIZE[resol]


def get_n_images(resol):
    return config.RESOL_N_IMAGES[resol]


def get_n_steps(n_images, batch_size):
    n_steps = n_images // batch_size
    return n_steps


def get_alpha(step, n_steps, trans_phase):
    if trans_phase:
        # "When doubling the resolution of the generator and discriminator we fade in the new layers smoothly.
        # During the transition we treat the layers that operate on the higher resolution like a residual block,
        # whose weight increases linearly from 0 to 1."
        alpha = step / n_steps
    else:
        alpha = 1
    return alpha


def get_dataloader(split, batch_size, resol):
    ds = CelebAHQDataset(data_dir=config.DATA_DIR, split=split, resol=resol)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=config.N_WORKERS, pin_memory=True, drop_last=True
    )
    return dl


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_model(model):
    for p in model.parameters():
        p.requires_grad = True


gen = Generator().to(DEVICE)
gen = nn.DataParallel(gen)

disc = Discriminator().to(DEVICE)
disc = nn.DataParallel(disc)

# "We train the networks using Adam with $\alpha = 0.001$, $\beta_{1} = 0$, $\beta_{2} = 0.99$,
# and $\epsilon = 10^{-8}$. We do not use any learning rate decay or rampdown, but for visualizing
# generator output at any given point during the training, we use an exponential running average
# for the weights of the generator with decay $0.999$."
gen_optim = Adam(params=gen.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2), eps=config.ADAM_EPS)
disc_optim = Adam(params=disc.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2), eps=config.ADAM_EPS)

gen_scaler = GradScaler()
disc_scaler = GradScaler()

### Resume from checkpoint.
if config.CKPT_PATH is not None:
    ckpt = torch.load(config.CKPT_PATH, map_location=DEVICE)
    disc.load_state_dict(ckpt["D"])
    gen.load_state_dict(ckpt["G"])
    disc_optim.load_state_dict(ckpt["D_optimizer"])
    gen_optim.load_state_dict(ckpt["G_optimizer"])

step = config.STEP if config.STEP is not None else ckpt["step"]
trans_phase = config.TRANS_PHASE if config.TRANS_PHASE is not None else ckpt["transition_phase"]
resol_idx = config.RESOL_IDX if config.RESOL_IDX is not None else ckpt["resolution_index"]
resol = RESOLS[resol_idx]
n_images = get_n_images(resol)
batch_size = get_batch_size(resol)
n_steps = get_n_steps(n_images=n_images, batch_size=batch_size)
if config.CKPT_PATH is not None:
    print(f"""Resuming from resolution {resol:,}×{resol:,} and step {step:,}/{n_steps:,}.""", end=" ")
    print(f"""(Transition phase: {trans_phase})""")

train_dl = get_dataloader(split="train", batch_size=batch_size, resol=resol)
train_di = iter(train_dl)

disc.train()
disc_running_loss = 0
gen_running_loss = 0
start_time = time()
while True:
    step += 1
    alpha = get_alpha(step=step, n_steps=n_steps, trans_phase=trans_phase)

    try:
        real_image = next(train_di)
    except StopIteration:
        train_di = iter(train_dl)
        real_image = next(train_di)
    real_image = real_image.to(DEVICE)

    gen.train()

    # "We alternate between optimizing the generator and discriminator on a per-minibatch basis."
    ### Optimize D.
    disc_optim.zero_grad()

    # "Our latent vectors correspond to random points on a 512-dimensional hypersphere."
    noise = torch.randn(batch_size, 512, 1, 1, device=DEVICE)
    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16) if config.AUTOCAST else nullcontext():
        real_pred = disc(real_image, resol=resol, alpha=alpha)
        fake_image = gen(noise, resol=resol, alpha=alpha)
        fake_pred = disc(fake_image.detach(), resol=resol, alpha=alpha)

    disc_loss1 = -torch.mean(real_pred) + torch.mean(fake_pred)
    gp = get_gradient_penalty(
        disc=disc, resol=resol, alpha=alpha, real_image=real_image, fake_image=fake_image.detach()
    )
    disc_loss2 = config.LAMBDA * gp
    # "We use the WGAN-GP loss."
    # "We introduce a fourth term into the discriminator loss with an extremely small weight
    # to keep the discriminator output from drifting too far away from zero. We set
    # $L' = L + \epsilon_{drift}\mathbb{E}_{x \in \mathbb{P}_{r}}[D(x)^{2}]$,
    # where $\epsilon_{drift} = 0.001$."
    disc_loss3 = config.LOSS_EPS * torch.mean(real_pred ** 2)
    disc_loss = disc_loss1 + disc_loss2 + disc_loss3
    if config.AUTOCAST:
        disc_scaler.scale(disc_loss).backward()
        disc_scaler.step(disc_optim)
        disc_scaler.update()
    else:
        disc_loss.backward()
        disc_optim.step()

    ### Optimize G.
    gen_optim.zero_grad()

    freeze_model(disc)
    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
        fake_pred = disc(fake_image, resol=resol, alpha=alpha)
        gen_loss = -torch.mean(fake_pred)
    if config.AUTOCAST:
        gen_scaler.scale(gen_loss).backward()
        gen_scaler.step(gen_optim)
        gen_scaler.update()
    else:
        gen_loss.backward()
        gen_optim.step()
    unfreeze_model(disc)

    disc_running_loss += disc_loss1.item()
    gen_running_loss += gen_loss.item()

    if (step % config.N_PRINT_STEPS == 0) or (step == n_steps):
        disc_running_loss /= config.N_PRINT_STEPS
        gen_running_loss /= config.N_PRINT_STEPS

        print(f"""[ {resol:,}×{resol:,} ][ {step:,}/{n_steps:,} ][ {alpha:.3f} ]""", end="")
        print(f"""[ {get_elapsed_time(start_time)} ]""", end="")
        print(f"""[ D loss: {disc_running_loss:.4f} ]""", end="")
        print(f"""[ G loss: {gen_running_loss:.4f} ]""", end="")
        print(f"""[ GP: {gp:.6f} ]""")
        start_time = time()

        gen.eval()
        with torch.no_grad():
            fake_image = gen(noise, resol=resol, alpha=alpha)
            fake_image = fake_image.detach().cpu()
            grid = image_to_grid(fake_image[: 9, ...], n_cols=3, value_range=(-1, 1))
            if trans_phase:
                save_path = IMG_DIR/f"""{resol // 2}×{resol // 2}to{resol}×{resol}/{step}.jpg"""
            else:
                save_path = IMG_DIR/f"""{resol}×{resol}/{step}.jpg"""
            save_image(grid, path=save_path)

        disc_running_loss = 0
        gen_running_loss = 0

    if (step % config.N_CKPT_STEPS == 0) or (step == n_steps):
        if trans_phase:
            filename = f"""{resol // 2}×{resol // 2}to{resol}×{resol}_{step}.pth"""
        else:
            filename = f"""{resol}×{resol}_{step}.pth"""
        save_checkpoint(
            resol_idx=resol_idx,
            step=step,
            trans_phase=trans_phase,
            disc=disc,
            gen=gen,
            disc_optim=disc_optim,
            gen_optim=gen_optim,
            save_path=CKPT_DIR/filename,
        )

    if step >= n_steps:
        if not trans_phase:
            resol_idx += 1
            resol = RESOLS[resol_idx]
            batch_size = get_batch_size(resol)
            n_images = get_n_images(resol)
            n_steps = get_n_steps(n_images=n_images, batch_size=batch_size)
            train_dl = get_dataloader(split="train", batch_size=batch_size, resol=resol)
        train_di = iter(train_dl)
        trans_phase = not trans_phase

        step = 0
