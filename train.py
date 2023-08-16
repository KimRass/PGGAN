# References:
    # https://github.com/ziwei-jiang/PGGAN-PyTorch/blob/master/train.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler
from pathlib import Path
from time import time
from contextlib import nullcontext

import config
from utils import (
    image_to_grid,
    save_image,
    get_elapsed_time,
    freeze_model,
    unfreeze_model,
    get_batch_size,
    get_n_images,
    get_n_steps,
    get_alpha,
)
from model import Generator, Discriminator
from celebahq import get_dataloader
from loss import get_gradient_penalty

print(f"""AUTOCAST = {config.AUTOCAST}""")
print(f"""N_WORKES = {config.N_WORKERS}""")

ROOT_DIR = Path(__file__).parent
CKPT_DIR = ROOT_DIR/"checkpoints"
IMG_DIR = ROOT_DIR/"generated_images"


def save_checkpoint(
    resol_idx, step, trans_phase, disc, gen, disc_optim, gen_optim, disc_scaler, gen_scaler, save_path
):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "resolution_index": resol_idx,
        "step": step,
        "transition_phase": trans_phase,
        "D_optimizer": disc_optim.state_dict(),
        "G_optimizer": gen_optim.state_dict(),
        "D_scaler": disc_scaler.state_dict(),
        "G_scaler": gen_scaler.state_dict(),
    }
    if config.N_GPUS > 1 and config.MULTI_GPU:
        ckpt["D"] = disc.module.state_dict()
        ckpt["G"] = gen.module.state_dict()
    else:
        ckpt["D"] = disc.state_dict()
        ckpt["G"] = gen.state_dict()

    torch.save(ckpt, str(save_path))


disc = Discriminator()
gen = Generator()
if config.N_GPUS > 0:
    DEVICE = torch.device("cuda")
    disc = disc.to(DEVICE)
    gen = gen.to(DEVICE)
    if config.N_GPUS > 1 and config.MULTI_GPU:
        disc = nn.DataParallel(disc)
        gen = nn.DataParallel(gen)

        print(f"""Using {config.N_GPUS} GPUs.""")
    else:
        print("Using single GPU.")
else:
    print("Using CPU.")

# "We train the networks using Adam with $\alpha = 0.001$, $\beta_{1} = 0$, $\beta_{2} = 0.99$,
# and $\epsilon = 10^{-8}$. We do not use any learning rate decay or rampdown, but for visualizing
# generator output at any given point during the training, we use an exponential running average
# for the weights of the generator with decay $0.999$."
disc_optim = Adam(
    params=disc.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2), eps=config.ADAM_EPS
)
gen_optim = Adam(
    params=gen.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2), eps=config.ADAM_EPS
)

disc_scaler = GradScaler()
gen_scaler = GradScaler()

### Resume from checkpoint.
if config.CKPT_PATH is not None:
    ckpt = torch.load(config.CKPT_PATH, map_location=DEVICE)
    disc.load_state_dict(ckpt["D"])
    gen.load_state_dict(ckpt["G"])
    disc_optim.load_state_dict(ckpt["D_optimizer"])
    gen_optim.load_state_dict(ckpt["G_optimizer"])
    disc_scaler.load_state_dict(ckpt["D_scaler"])
    gen_scaler.load_state_dict(ckpt["G_scaler"])

step = config.STEP if config.STEP is not None else ckpt["step"]
trans_phase = config.TRANS_PHASE if config.TRANS_PHASE is not None else ckpt["transition_phase"]
resol_idx = config.RESOL_IDX if config.RESOL_IDX is not None else ckpt["resolution_index"]
resol = config.RESOLS[resol_idx]
n_images = get_n_images(resol)
batch_size = get_batch_size(resol)
print(f"""batch_size = {batch_size}""")
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
    with torch.autocast(
        device_type=DEVICE.type, dtype=torch.float16
    ) if config.AUTOCAST else nullcontext():
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

    with torch.autocast(
        device_type=DEVICE.type, dtype=torch.float16
    ) if config.AUTOCAST else nullcontext():
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
        print(f"""[ D loss: {disc_running_loss:.3f} ]""", end="")
        print(f"""[ G loss: {gen_running_loss:.3f} ]""", end="")
        print(f"""[ GP: {gp:.3f} ]""")
        start_time = time()

        gen.eval()
        with torch.no_grad():
            fake_image = gen(noise, resol=resol, alpha=alpha)
            fake_image = fake_image.detach().cpu()
            grid = image_to_grid(fake_image[: 9, ...], n_cols=2 if resol == 1024 else 3, value_range=(-1, 1))
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
            disc_scaler=disc_scaler,
            gen_scaler=gen_scaler,
            save_path=CKPT_DIR/filename,
        )

    if step >= n_steps:
        if not trans_phase:
            resol_idx += 1
            resol = config.RESOLS[resol_idx]
            batch_size = get_batch_size(resol)
            print(f"""batch_size = {batch_size}""")
            n_images = get_n_images(resol)
            n_steps = get_n_steps(n_images=n_images, batch_size=batch_size)
            train_dl = get_dataloader(split="train", batch_size=batch_size, resol=resol)
        train_di = iter(train_dl)
        trans_phase = not trans_phase

        step = 0
