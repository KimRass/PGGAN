# References:
    # https://www.kaggle.com/code/heonh0/pggan-progressive-growing-gan-pggan-pytorch

import torch
from torch.optim import Adam
# from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

from torch_utils import get_device, save_parameters, batched_image_to_grid
from image_utils import save_image
from model import Generator, Discriminator
from celebahq import CelebAHQDataset
from loss import get_gradient_penalty

R2B = {4: 16, 8: 16, 16: 16, 32: 16, 64: 16, 128: 16, 256: 14, 512: 6, 1024: 3}
# "We start with $4 \times 4$ resolution and train the networks until we have shown the discriminator
# 800k real images in total. We then alternate between two phases: fade in the first 3-layer block
# during the next 800k images, stabilize the networks for 800k images, fade in the next 3-layer block
# during 800k images, etc."
N_ITERS = 800_000
# "When doubling the resolution of the generator and discriminator we fade in the new layers smoothly.
# During the transition we treat the layers that operate on the higher resolution like a residual block,
# whose weight increases linearly from 0 to 1."
ALPHAS = np.linspace(0, 1, N_ITERS)

# "We use a minibatch size $16$ for resolutions $4^{2}$–$128^{2}$ and then gradually decrease
# the size according to $256^{2} \rightarrow 14$, $512^{2} \rightarrow 6$, $1024^{2} \rightarrow 3$
# to avoid exceeding the available memory budget."
def get_batch_size(resol):
    return R2B[resol]


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
ROOT = "/home/ubuntu/project/celebahq/celeba_hq"
# ROOT = "/Users/jongbeomkim/Documents/datasets/celebahq/"
ds = CelebAHQDataset(root=ROOT, split="train", resol=RESOLS[0])
TRANS_PHASE = False
res_idx = 0
resol = RESOLS[res_idx]
batch_size = get_batch_size(resol)
N_WORKERS = 4
# N_WORKERS = 0
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=N_WORKERS, drop_last=True)
LAMBDA = 10
EPS = 0.001

iter_ = 0
breaker = False
while True:
    if breaker:
        break

    for batch, real_image in enumerate(dl, start=1):
        iter_ += 1
        if TRANS_PHASE:
            alpha = get_alpha(iter_)
        else:
            alpha = 1

        real_image = real_image.to(DEVICE).detach()
        ### Optimize D.
        # G와 D 중 어느 것을 먼저 학습시키는지가 중요한지는 잘 모르겠지만 다른 코드에서는 보통 D를 먼저 학습시키는 듯합니다.
        # "Our latent vectors correspond to random points on a 512-dimensional hypersphere."
        noise = torch.randn(batch_size, 512, 1, 1, device=DEVICE)
        with torch.autocast(device_type=DEVICE.type):
            gen_image = gen(noise, resol=resol, alpha=alpha).detach()
            real_pred = disc(real_image, resol=resol, alpha=alpha)
            gen_pred = disc(gen_image, resol=resol, alpha=alpha)

            disc_loss = -torch.mean(real_pred) + torch.mean(gen_pred)
            gp = get_gradient_penalty(
                disc=disc, resol=resol, alpha=alpha, real_image=real_image, gen_image=gen_image
            )
            disc_loss += LAMBDA * gp
            # "We introduce a fourth term into the discriminator loss with an extremely small weight
            # to keep the discriminator output from drifting too far away from zero. We set
            # $L' = L + \epsilon_{drift}\mathbb{E}_{x \in \mathbb{P}_{r}}[D(x)^{2}]$,
            # where $\epsilon_{drift} = 0.001$."
            disc_loss += EPS * torch.mean(real_pred ** 2)

        disc_optim.zero_grad()
        disc_scaler.scale(disc_loss).backward()
        disc_scaler.step(disc_optim)
        disc_scaler.update()

        ### Optimize G.
        with torch.autocast(device_type=DEVICE.type):
            gen_pred = disc(gen_image, resol=resol, alpha=alpha)
            gen_loss = -torch.mean(gen_pred)

        gen_optim.zero_grad()
        # "We use the WGAN-GP loss. We alternate between optimizing the generator and discriminator
        # on a per-minibatch basis, i.e., we set $n_{critic} = 1$."
        gen_scaler.scale(gen_loss).backward()
        gen_scaler.step(gen_optim)
        gen_scaler.update()

        if iter_ % (N_ITERS // 1000) == 0:
            print(f"""[ {resol} ][ {iter_}/{N_ITERS} ][ {alpha} ]""", end=" ")
            print(f"""G loss: {gen_loss.item(): .0f} | D loss: {disc_loss.item(): .0f}""")

            gen_image = gen_image.detach().cpu()
            grid = batched_image_to_grid(
                gen_image[: 3, ...], n_cols=3, mean=(0.517, 0.416, 0.363), std=(0.303, 0.275, 0.269)
            )
            root_dir = Path(__file__).parent
            save_image(
                grid, path=root_dir/f"""generated_images/resol_{resol}_iter_{iter_}_alpha_{alpha}.jpg"""
            )

            save_parameters(
                model=gen,
                save_path=root_dir/f"""pretrained/resol_{resol}_iter_{iter_}_alpha_{alpha}.pth"""
            )

        if iter_ == N_ITERS:
            if resol == RESOLS[-1] and not TRANS_PHASE:
                breaker = True
                break
            elif not TRANS_PHASE:
                res_idx += 1
                resol = RESOLS[res_idx]
                batch_size = get_batch_size(resol)
                ds = CelebAHQDataset(root=ROOT, split="train", resol=resol)
                dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=N_WORKERS, drop_last=True)
                TRANS_PHASE = True
                iter_ = 0
            else:
                TRANS_PHASE = False
                iter_ = 0
