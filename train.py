# References:
    # https://github.com/ziwei-jiang/PGGAN-PyTorch/blob/master/train.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler
from pathlib import Path
from time import time
from tqdm.auto import tqdm

import config
from utils import (
    image_to_grid,
    save_image,
    get_elapsed_time,
    freeze_model,
    unfreeze_model,
)
from model import Generator, Discriminator
from celebahq import get_dataloader
from loss import get_gradient_penalty
from eval import get_swd


# "We use a minibatch size $16$ for resolutions $4^{2}$–$128^{2}$ and then gradually decrease
# the size according to $256^{2} \rightarrow 14$, $512^{2} \rightarrow 6$, $1024^{2} \rightarrow 3$
# to avoid exceeding the available memory budget."
def get_batch_size(img_size):
    return config.IMG_SIZE_BATCH_SIZE[img_size]


def get_n_images(img_size):
    return config.IMG_SIZE_N_IMAGES[img_size]


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


@torch.no_grad()
def validate(G, val_dl, device):
    print(f"Validating...")
    G.eval()
    sum_swd = 0
    for real_image in tqdm(val_dl, leave=False):
        real_image = real_image.to(device)
        noise = torch.randn(batch_size, 512, 1, 1, device=device)
        fake_image = G(noise, img_size=img_size, alpha=alpha)

        swd = get_swd(real_image, fake_image, device=device)
        sum_swd += swd.item()
    avg_swd = sum_swd / len(val_dl)
    print(f"Average SWD: {avg_swd: .3f}")
    G.train()
    return avg_swd


def save_checkpoint(
    img_size_idx,
    step,
    trans_phase,
    D,
    G,
    D_optim,
    G_optim,
    scaler,
    avg_swd,
    save_path,
):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "image_size_index": img_size_idx,
        "step": step,
        "transition_phase": trans_phase,
        "D_optimizer": D_optim.state_dict(),
        "G_optimizer": G_optim.state_dict(),
        "scaler": scaler.state_dict(),
        "average_swd": avg_swd,
    }
    if config.N_GPUS > 1 and config.MULTI_GPU:
        ckpt["D"] = D.module.state_dict()
        ckpt["G"] = G.module.state_dict()
    else:
        ckpt["D"] = D.state_dict()
        ckpt["G"] = G.state_dict()

    torch.save(ckpt, str(save_path))


if __name__ == "__main__":
    print(f"AMP = {config.AMP}")
    print(f"N_WORKERS = {config.N_WORKERS}")

    ROOT_DIR = Path(__file__).parent
    CKPT_DIR = ROOT_DIR/"checkpoints"
    IMG_DIR = ROOT_DIR/"generated_images"

    D = Discriminator()
    G = Generator()
    if config.N_GPUS > 0:
        DEVICE = torch.device("cuda")
        D = D.to(DEVICE)
        G = G.to(DEVICE)
        if config.N_GPUS > 1 and config.MULTI_GPU:
            D = nn.DataParallel(D)
            G = nn.DataParallel(G)

            print(f"Using {config.N_GPUS} GPUs.")
        else:
            print("Using single GPU.")
    else:
        print("Using CPU.")

    # "We train the networks using Adam with $\alpha = 0.001$, $\beta_{1} = 0$, $\beta_{2} = 0.99$,
    # and $\epsilon = 10^{-8}$. We do not use any learning rate decay or rampdown, but for visualizing
    # generator output at any given point during the training, we use an exponential running average
    # for the weights of the generator with decay $0.999$."
    D_optim = Adam(
        params=D.parameters(),
        lr=config.LR,
        betas=(config.BETA1, config.BETA2),
        eps=config.ADAM_EPS,
    )
    G_optim = Adam(
        params=G.parameters(),
        lr=config.LR,
        betas=(config.BETA1, config.BETA2),
        eps=config.ADAM_EPS,
    )

    scaler = GradScaler()

    ### Resume from checkpoint.
    if config.CKPT_PATH is not None:
        ckpt = torch.load(config.CKPT_PATH, map_location=DEVICE)

        if config.N_GPUS > 1 and config.MULTI_GPU:
            D.module.load_state_dict(ckpt["D"])
            G.module.load_state_dict(ckpt["G"])
        else:
            D.load_state_dict(ckpt["D"])
            G.load_state_dict(ckpt["G"])
        D_optim.load_state_dict(ckpt["D_optimizer"])
        G_optim.load_state_dict(ckpt["G_optimizer"])
        scaler.load_state_dict(ckpt["scaler"])

        if "average_swd" in ckpt:
            best_avg_swd = ckpt["average_swd"]
            prev_save_path = config.CKPT_PATH
        else:
            best_avg_swd = 0
            prev_save_path = ".pth"
    else:
        best_avg_swd = 0
        prev_save_path = ".pth"

    step = config.STEP if config.STEP is not None else ckpt["step"]
    trans_phase = config.TRANS_PHASE if config.TRANS_PHASE is not None else ckpt["transition_phase"]
    img_size_idx = config.IMG_SIZE_IDX if config.IMG_SIZE_IDX is not None else ckpt["image_size_index"]
    img_size = config.IMG_SIZES[img_size_idx]
    n_images = get_n_images(img_size)
    batch_size = get_batch_size(img_size)
    print(f"batch_size = {batch_size}")
    n_steps = get_n_steps(n_images=n_images, batch_size=batch_size)
    if config.CKPT_PATH is not None:
        print(f"Resuming from resolution {img_size:,}×{img_size:,}", end="")
        print(f" and step {step:,}/{n_steps:,}.", end="")
        print(f" (Transition phase: {trans_phase})")

    train_dl = get_dataloader(split="train", batch_size=batch_size, img_size=img_size)
    train_di = iter(train_dl)
    val_dl = get_dataloader(split="val", batch_size=batch_size, img_size=img_size)

    D_running_loss = 0
    G_running_loss = 0
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

        # "We alternate between optimizing the generator and discriminator on a per-minibatch basis."
        ### Optimize D.
        # "Our latent vectors correspond to random points on a 512-dimensional hypersphere."
        noise = torch.randn(batch_size, 512, 1, 1, device=DEVICE)
        with torch.autocast(
            device_type=DEVICE.type, dtype=torch.float16, enabled=True if config.AMP else False
        ):
            real_pred = D(real_image, img_size=img_size, alpha=alpha)
            fake_image = G(noise, img_size=img_size, alpha=alpha)
            fake_pred = D(fake_image.detach(), img_size=img_size, alpha=alpha)

            D_loss1 = -torch.mean(real_pred) + torch.mean(fake_pred)

            gp = get_gradient_penalty(
                D=D,
                img_size=img_size,
                alpha=alpha,
                real_image=real_image,
                fake_image=fake_image.detach(),
            )
            D_loss2 = config.LAMBDA * gp

            # "We use the WGAN-GP loss."
            # "We introduce a fourth term into the discriminator loss with an extremely small weight
            # to keep the discriminator output from drifting too far away from zero. We set
            # $L' = L + \epsilon_{drift}\mathbb{E}_{x \in \mathbb{P}_{r}}[D(x)^{2}]$,
            # where $\epsilon_{drift} = 0.001$."
            D_loss3 = config.LOSS_EPS * torch.mean(real_pred ** 2)
            D_loss = D_loss1 + D_loss2 + D_loss3

        D_optim.zero_grad()
        if config.AMP:
            scaler.scale(D_loss).backward()
            scaler.step(D_optim)
        else:
            D_loss.backward()
            D_optim.step()

        ### Optimize G.
        freeze_model(D)

        with torch.autocast(
            device_type=DEVICE.type, dtype=torch.float16, enabled=True if config.AMP else False
        ):
            fake_pred = D(fake_image, img_size=img_size, alpha=alpha)
            G_loss = -torch.mean(fake_pred)

        G_optim.zero_grad()
        if config.AMP:
            scaler.scale(G_loss).backward()
            scaler.step(G_optim)
        else:
            G_loss.backward()
            G_optim.step()

        unfreeze_model(D)

        if config.AMP:
            scaler.update()

        D_running_loss += D_loss1.item()
        G_running_loss += G_loss.item()

        if (step % config.N_PRINT_STEPS == 0) or (step == n_steps):
            D_running_loss /= config.N_PRINT_STEPS
            G_running_loss /= config.N_PRINT_STEPS

            print(f"[ {img_size:,}×{img_size:,} ][ {step:,}/{n_steps:,} ][ {alpha:.3f} ]", end="")
            print(f"[ {get_elapsed_time(start_time)} ]", end="")
            print(f"[ D loss: {D_running_loss:.3f} ]", end="")
            print(f"[ G loss: {G_running_loss:.3f} ]", end="")
            print(f"[ GP: {gp.item():.3f} ]")
            start_time = time()

            G.eval()
            with torch.no_grad():
                fake_image = G(noise, img_size=img_size, alpha=alpha)
                fake_image = fake_image.detach().cpu()
                grid = image_to_grid(
                    fake_image[: 9, ...], n_cols=2 if img_size == 1024 else 3, value_range=(-1, 1)
                )
                if trans_phase:
                    save_path = IMG_DIR/f"{img_size // 2}×{img_size // 2}to{img_size}×{img_size}/{step}.jpg"
                else:
                    save_path = IMG_DIR/f"{img_size}×{img_size}/{step}.jpg"
                save_image(grid, path=save_path)
            G.train()

            D_running_loss = 0
            G_running_loss = 0

        if (step % config.N_VAL_STEPS == 0) or (step == n_steps):
            avg_swd = validate(G=G, val_dl=val_dl, device=DEVICE)
            if avg_swd < best_avg_swd:
                if trans_phase:
                    cur_save_path = CKPT_DIR/\
                        f"{img_size // 2}×{img_size // 2}to{img_size}×{img_size}_{step}.pth"
                else:
                    cur_save_path = CKPT_DIR/f"{img_size}×{img_size}_{step}.pth"
                save_checkpoint(
                    img_size_idx=img_size_idx,
                    step=step,
                    trans_phase=trans_phase,
                    avg_swd=avg_swd,
                    D=D,
                    G=G,
                    D_optim=D_optim,
                    G_optim=G_optim,
                    scaler=scaler,
                    save_path=cur_save_path,
                )
                prev_save_path = Path(prev_save_path)
                if prev_save_path.exists():
                    prev_save_path.unlink()
                print(f"Saved checkpoint.")

                best_avg_swd = avg_swd
                prev_save_path = cur_save_path

        if step >= n_steps:
            if not trans_phase:
                img_size_idx += 1
                img_size = config.IMG_SIZES[img_size_idx]
                batch_size = get_batch_size(img_size)
                print(f"batch_size = {batch_size}")
                n_images = get_n_images(img_size)
                n_steps = get_n_steps(n_images=n_images, batch_size=batch_size)
                train_dl = get_dataloader(split="train", batch_size=batch_size, img_size=img_size)
            train_di = iter(train_dl)
            trans_phase = not trans_phase

            step = 0
            best_avg_swd = 0
