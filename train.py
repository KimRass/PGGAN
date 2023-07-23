# References:
    # https://www.kaggle.com/code/heonh0/pggan-progressive-growing-gan-pggan-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

# "When doubling the resolution of the generator and discriminator we fade in the new layers smoothly.
# During the transition we treat the layers that operate on the higher resolution like a residual block,
# whose weight increases linearly from 0 to 1." 
# When training the discriminator, we feed in real images that are downscaled to match the current resolution of the network. During a resolution transition, we interpolate between two resolutions of the real images, similarly to how the generator output combines two resolutions.
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

# "We start with $4 \times 4$ resolution and train the networks until we have shown the discriminator
# 800k real images in total. We then alternate between two phases: fade in the first 3-layer block
# during the next 800k images, stabilize the networks for 800k images, fade in the next 3-layer block
# during 800k images, etc."
# 800_000 / (10_057 + 17_943)

# "Our latent vectors correspond to random points on a 512-dimensional hypersphere."