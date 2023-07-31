# References:
    # https://github.com/ziwei-jiang/PGGAN-PyTorch/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from utils import print_number_of_parameters

# "We use leaky ReLU with leakiness 0.2 in all layers of both networks, except for the last layer
# that uses linear activation."
LEAKINESS = 0.2
GAIN = 0.2


# "EQUALIZED LEARNING RATE: W use a trivial $\mathcal{N}(0, 1)$ initialization and then
# explicitly scale the weights at runtime. We set $w^{^}_{i} = w_{i} / c$, where $w_{i}$ are the weights
# and $c$ is the per-layer normalization constant from He’s initializer."
# "We initialize all bias parameters to zero and all weights according to the normal distribution
# with unit variance. However, we scale the weights with a layer-specific constant at runtime."
# The idea is to scale the parameters of each layer just before every forward propagation
# that passes through. How much to scale by is determined by a statistic calculated
# from the parameter values of each layer.
class EqualLRLinear(nn.Module):
    def __init__(self, in_features, out_features, gain=GAIN):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.gain = gain

        self.scale = np.sqrt(gain / in_features)

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)
 
    def forward(self, x):
        x = F.linear(x, weight=self.weight * self.scale, bias=self.bias)
        return x


class EqualLRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, gain=GAIN):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.gain = gain

        self.scale = (gain / (in_channels * kernel_size * kernel_size)) ** 0.5

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)
                
    def forward(self, x):
        x = F.conv2d(x, weight=self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
        return x


# "The `toRGB` represents a layer that projects feature vectors to RGB colors. It uses 1×1 convolutions."
class ToRGB(nn.Module):
    def __init__(self, in_channels, leakiness=LEAKINESS):
        super().__init__()

        self.leakiness = leakiness

        self.in_channels = in_channels
        self.conv = EqualLRConv2d(in_channels, 3, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


# "The `fromRGB` does the reverse of `toRGB`. it uses 1×1 convolutions."
class FromRGB(nn.Module):
    def __init__(self, out_channels, leakiness=LEAKINESS):
        super().__init__()

        self.leakiness = leakiness

        self.out_channels = out_channels
        self.conv = EqualLRConv2d(3, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(x, negative_slope=self.leakiness)
        return x

# "PIXELWISE FEATURE VECTOR NORMALIZATION IN GENERATOR: We normalize the feature vector in each pixel
# to unit length in the generator after each convolutional layer."
# "$b_{x, y} = a_{x, y} / \sqrt{1 / N \sum^{N - 1}_{j=0}(a^{j}_{x, y})^{2} + \epsilon}$, where
# $\epsilon = 10^{-8}$, $N$ is the number of feature maps, and $a_{x, y}$ and $b_{x, y}$ are
# the original and normalized feature vector in pixel $(x, y)$, respectively."
def perform_pixel_norm(x, eps=1e-8):
    x = x / torch.sqrt((x ** 2).mean(dim=1, keepdim=True)+ eps)
    return x


# "'2×' refer to doubling the image resolution using nearest neighbor filtering."
def _double(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True, leakiness=LEAKINESS):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        self.leakiness = leakiness

        if upsample:
            self.conv1 = EqualLRConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.conv1 = EqualLRConv2d(in_channels, out_channels, kernel_size=4, padding=3)
        self.conv2 = EqualLRConv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        ### ORDER OF LAYERS!
        if self.upsample:
            x = _double(x)
        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=self.leakiness)
        # "We perform pixel-wise normalization of the feature vectors after each Conv 3×3 layer
        # in the generator."
        x = perform_pixel_norm(x)
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=self.leakiness)
        x = perform_pixel_norm(x)
        return x


def _get_depth(resol):
    depth = int(math.log2(resol)) - 1
    return depth


class Generator(nn.Module): # 23,079,115 ("23.1M") parameters in total.
    def __init__(self):
        super().__init__()

        self.block1 = UpsampleBlock(512, 512, upsample=False)
        self.block2 = UpsampleBlock(512, 512)
        self.block3 = UpsampleBlock(512, 512)
        self.block4 = UpsampleBlock(512, 512)
        self.block5 = UpsampleBlock(512, 256)
        self.block6 = UpsampleBlock(256, 128)
        self.block7 = UpsampleBlock(128, 64)
        self.block8 = UpsampleBlock(64, 32)
        self.block9 = UpsampleBlock(32, 16)

        # "The last Conv 1×1 layer of the generator corresponds to the 'toRGB' block."
        self.to_rgb1 = ToRGB(512)
        self.to_rgb2 = ToRGB(512)
        self.to_rgb3 = ToRGB(512)
        self.to_rgb4 = ToRGB(512)
        self.to_rgb5 = ToRGB(256)
        self.to_rgb6 = ToRGB(128)
        self.to_rgb7 = ToRGB(64)
        self.to_rgb8 = ToRGB(32)
        self.to_rgb9 = ToRGB(16)

    def forward(self, x, resol, alpha):
        if resol == 4:
            x = self.block1(x)
            x = self.to_rgb1(x)
        else:
            depth = _get_depth(resol)
            for d in range(1, depth):
                x = eval(f"""self.block{d}""")(x)

            skip = x.clone()
            skip = _double(skip)
            skip = eval(f"""self.to_rgb{depth - 1}""")(skip)

            x = eval(f"""self.block{depth}""")(x)
            x = eval(f"""self.to_rgb{depth}""")(x)

            x = (1 - alpha) * skip + alpha * x
        return x


# "'0.5×' refer to halving the image resolution using nearest neighbor average pooling."
def _half(x):
    return F.avg_pool2d(x, kernel_size=2, stride=2)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True, leakiness=LEAKINESS):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.leakiness = leakiness

        if downsample:
            self.conv1 = EqualLRConv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = EqualLRConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.conv1 = EqualLRConv2d(in_channels + 1, out_channels, kernel_size=3, padding=1)
            self.conv2 = EqualLRConv2d(out_channels, out_channels, kernel_size=4)
            self.proj = EqualLRLinear(out_channels, 1)

    def add_minibatch_std(self, x):
        b, _, h, w = x.shape
        # "We compute the standard deviation for each feature in each spatial location over the minibatch.
        # We then average these estimates over all features and spatial locations to arrive at a single value.
        # We replicate the value and concatenate it to all spatial locations and over the minibatch,
        # yielding one additional (constant) feature map."
        feat_map = x.std(dim=0, keepdim=True).mean(dim=(1, 2, 3), keepdim=True)
        x = torch.cat([x, feat_map.repeat(b, 1, h, w)], dim=1)
        return x

    def forward(self, x):
        if not self.downsample:
            # "We inject the across-minibatch standard deviation as an additional feature map
            # at 4×4 resolution toward the end of the discriminator."
            x = self.add_minibatch_std(x)
        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=self.leakiness)
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=self.leakiness)
        if self.downsample:
            x = _half(x)
        else:
            x = x.view(-1, self.out_channels)
            x = self.proj(x)
            x = x.view(-1, 1, 1, 1)
        return x


class Discriminator(nn.Module): # 25,444,737 parameters in total.
    def __init__(self):
        super().__init__()

        self.block1 = DownsampleBlock(512, 512, downsample=False)
        self.block2 = DownsampleBlock(512, 512)
        self.block7 = DownsampleBlock(64, 128)
        self.block4 = DownsampleBlock(512, 512)
        self.block5 = DownsampleBlock(256, 512)
        self.block6 = DownsampleBlock(128, 256)
        self.block8 = DownsampleBlock(32, 64)
        self.block3 = DownsampleBlock(512, 512)
        self.block9 = DownsampleBlock(16, 32)

        self.from_rgb1 = FromRGB(512)
        self.from_rgb2 = FromRGB(512)
        self.from_rgb3 = FromRGB(512)
        self.from_rgb4 = FromRGB(512)
        self.from_rgb5 = FromRGB(256)
        self.from_rgb6 = FromRGB(128)
        self.from_rgb7 = FromRGB(64)
        self.from_rgb8 = FromRGB(32)
        self.from_rgb9 = FromRGB(16)

    def forward(self, x, resol, alpha):
        if resol == 4:
            x = self.from_rgb1(x)
            x = self.block1(x)
        else:
            depth = _get_depth(resol)

            skip = x.clone()
            skip = _half(skip)
            skip = eval(f"""self.from_rgb{depth - 1}""")(skip)
            
            x = eval(f"""self.from_rgb{depth}""")(x)
            x = eval(f"""self.block{depth}""")(x)

            x = (1 - alpha) * skip + alpha * x

            for d in range(depth - 1, 0, -1):
                x = eval(f"""self.block{d}""")(x)
        return x


if __name__ == "__main__":
    BATCH_SIZE = 2
    for resol in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        alpha = 0.5
        gen = Generator()
        x = torch.randn(BATCH_SIZE, 512, 1, 1)
        out = gen(x, resol=resol, alpha=alpha)
        out.shape

        disc = Discriminator()
        x = torch.randn((BATCH_SIZE, 3, resol, resol))
        out = disc(x, resol=resol, alpha=alpha)
        out.shape
