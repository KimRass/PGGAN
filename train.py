# References:
    # https://www.kaggle.com/code/heonh0/pggan-progressive-growing-gan-pggan-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from pathlib import Path


# When training the discriminator, we feed in real images that are downscaled to match the current resolution
# of the network."

# "We start with $4 \times 4$ resolution and train the networks until we have shown the discriminator
# 800k real images in total. We then alternate between two phases: fade in the first 3-layer block
# during the next 800k images, stabilize the networks for 800k images, fade in the next 3-layer block
# during 800k images, etc."

# "GANs are prone to the escalation of signal magnitudes as a result of unhealthy competition
# between the two networks. We believe that the actual need in GANs is constraining signal magnitudes
# and competition. We use a different approach that consists of two ingredients, neither of which
# include learnable parameters."

# "EQUALIZED LEARNING RATE: W use a trivial $\mathcal{N}(0, 1)$ initialization and then
# explicitly scale the weights at runtime. We set $w^{^}_{i} = w_{i} / c$, where $w_{i}$ are the weights
# and $c$ is the per-layer normalization constant from He’s initializer."

# 800_000 / (10_057 + 17_943)

# "Our latent vectors correspond to random points on a 512-dimensional hypersphere."

from model import Generator, Discriminator
from celebahq import CelebAHQDataset
from torch_utils import get_device

DEVICE = get_device()
gen = Generator().to(DEVICE)
disc = Discriminator().to(DEVICE)

N_ITERS = 800_000
RESOLUTIONS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
ds = CelebAHQDataset(root="/Users/jongbeomkim/Documents/datasets/celebahq", split="train", resolution=4)

res_idx = 0
resolution = RESOLUTIONS[res_idx]
iter = 0
breaker = False
while True:
    if breaker:
        break

    for batch, image in enumerate(dl, start=1):
        iter += 1

        image = image.to(DEVICE)

        errD = -torch.mean(netD(real_img)) + torch.mean(netD(fake_img)) # Wasserstein
        errG = -torch.mean(netD(gen_img))

        if iter == N_ITERS:
            print(resolution, iter)
            if resolution == RESOLUTIONS[-1]:
                breaker = True
                break
            else:
                res_idx += 1
                resolution = RESOLUTIONS[res_idx]
            iter = 0

