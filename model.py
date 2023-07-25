# References:
    # https://github.com/nashory/pggan-pytorch/blob/master/network.py
    # https://github.com/ziwei-jiang/PGGAN-PyTorch/blob/master/model.py
    # https://personal-record.onrender.com/post/equalized-lr/

import torch
import torch.nn as nn
import torch.nn.functional as F


# "The `toRGB` represents a layer that projects feature vectors to RGB colors. It uses $1 \times 1$ convolutions."
class ToRGB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


# "The `fromRGB` does the reverse of `toRGB`. it uses $1 \times 1$ convolutions."
class FromRGB(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

# "PIXELWISE FEATURE VECTOR NORMALIZATION IN GENERATOR: We normalize the feature vector in each pixel
# to unit length in the generator after each convolutional layer."
# "$b_{x, y} = a_{x, y} / \sqrt{1 / N \sum^{N - 1}_{j=0}(a^{j}_{x, y})^{2} + \epsilon}$, where
# $\epsilon = 10^{-8}$, $N$ is the number of feature maps, and $a_{x, y}$ and $b_{x, y}$ are
# the original and normalized feature vector in pixel $(x, y)$, respectively."
def perform_pixel_norm(x, eps=1e-8):
    x = x / torch.sqrt((x ** 2).mean(dim=1, keepdim=True)+ eps)
    return x


class GFirstConvBlock(nn.Module):
    def __init__(self, leakiness=0.2):
        super().__init__()

        self.leakiness = leakiness

        self.conv1 = nn.Conv2d(512, 512, kernel_size=4, padding=3)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

    def forward(self, x):
        ### ORDER OF LAYERS!
        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=self.leakiness)
        x = perform_pixel_norm(x)
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=self.leakiness)
        x = perform_pixel_norm(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, leakiness=0.2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.leakiness = leakiness

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        ### ORDER OF LAYERS!
        # "'$2\times$' refer to doubling the image resolution using nearest neighbor filtering."
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        # "We use leaky ReLU with leakiness 0.2 in all layers of both networks, except for the last layer
        # that uses linear activation."
        x = F.leaky_relu(x, negative_slope=self.leakiness)
        # "We perform pixel-wise normalization of the feature vectors after each Conv $3 \times 3$ layer
        # in the generator."
        x = perform_pixel_norm(x)
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=self.leakiness)
        x = perform_pixel_norm(x)
        return x


# "We initialize all bias parameters to zero and all weights according to the normal distribution
# with unit variance. However, we scale the weights with a layer-specific constant at runtime."
# The idea is to scale the parameters of each layer just before every forward propagation
# that passes through. How much to scale by is determined by a statistic calculated
# from the parameter values of each layer.
def _init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            m.weight.data.normal_(0, 1)
            m.bias.data.zero_()


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = GFirstConvBlock()
        self.block2 = UpsampleBlock(in_channels=512, out_channels=512)
        self.block3 = UpsampleBlock(in_channels=512, out_channels=512)
        self.block4 = UpsampleBlock(in_channels=512, out_channels=512)
        self.block5 = UpsampleBlock(in_channels=512, out_channels=256)
        self.block6 = UpsampleBlock(in_channels=256, out_channels=128)
        self.block7 = UpsampleBlock(in_channels=128, out_channels=64)
        self.block8 = UpsampleBlock(in_channels=64, out_channels=32)
        self.block9 = UpsampleBlock(in_channels=32, out_channels=16)

        # "The last Conv $1 \times 1$ layer of the generator corresponds to the 'toRGB' block."
        self.to_rgb_512 = ToRGB(in_channels=512)
        self.to_rgb_256 = ToRGB(in_channels=256)
        self.to_rgb_128 = ToRGB(in_channels=128)
        self.to_rgb_64 = ToRGB(in_channels=64)
        self.to_rgb_32 = ToRGB(in_channels=32)
        self.to_rgb_16 = ToRGB(in_channels=16)

        _init_weights(self)

    def _fade_in(self, x, block, alpha):
        def _double(x):
            return F.interpolate(x, scale_factor=2, mode="nearest")

        def _forward_through_rgb_layer(x):
            return eval(f"""self.to_rgb_{x.shape[1]}""")(x)

        skip = x.clone()
        skip = block(skip)
        skip = _forward_through_rgb_layer(skip)

        x = _double(x)
        x = _forward_through_rgb_layer(x)
        return (1 - alpha) * x + alpha * skip

    def forward(self, x, resol, alpha):
        x = self.block1(x) # `(b, 512, 4, 4)`
        if resol == 4:
            x = self.to_rgb_512(x)
            return x
        elif resol == 8:
            x = self._fade_in(x=x, block=self.block2, alpha=alpha)
            return x
        if resol >= 16:
            x = self.block2(x) # `(b, 512, 8, 8)`
            if resol == 16:
                x = self._fade_in(x=x, block=self.block3, alpha=alpha)
                return x
        if resol >= 32:
            x = self.block3(x) # `(b, 512, 16, 16)`
            if resol == 32:
                x = self._fade_in(x=x, block=self.block4, alpha=alpha)
                return x
        if resol >= 64:
            x = self.block4(x) # `(b, 512, 32, 32)`
            if resol == 64:
                x = self._fade_in(x=x, block=self.block5, alpha=alpha)
                return x
        if resol >= 128:
            x = self.block5(x) # `(b, 256, 64, 64)`
            if resol == 128:
                x = self._fade_in(x=x, block=self.block6, alpha=alpha)
                return x
        if resol >= 256:
            x = self.block6(x) # `(b, 128, 128, 128)`
            if resol == 256:
                x = self._fade_in(x=x, block=self.block7, alpha=alpha)
                return x
        if resol >= 512:
            x = self.block7(x) # `(b, 64, 256, 256)`
            if resol == 512:
                x = self._fade_in(x=x, block=self.block8, alpha=alpha)
                return x
        if resol >= 1024:
            x = self.block8(x) # `(b, 32, 512, 512)`
            if resol == 1024:
                x = self._fade_in(x=x, block=self.block9, alpha=alpha)
                return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, leakiness=0.2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.leakiness = leakiness

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=self.leakiness)
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=self.leakiness)
        x = F.interpolate(x, scale_factor=0.5, mode="nearest")
        return x


class DLastConvBlock(nn.Module):
    def __init__(self, leakiness=0.2):
        super().__init__()

        self.leakiness = leakiness

        self.conv1 = nn.Conv2d(513, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=4)
        self.proj = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=self.leakiness)
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=self.leakiness)
        x = x.view(-1, 512)
        x = self.proj(x)
        x = x.view(-1, 1, 1, 1)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = DownsampleBlock(16, 32)
        self.block2 = DownsampleBlock(32, 64)
        self.block3 = DownsampleBlock(64, 128)
        self.block4 = DownsampleBlock(128, 256)
        self.block5 = DownsampleBlock(256, 512)
        self.block6 = DownsampleBlock(512, 512)
        self.block7 = DownsampleBlock(512, 512)
        self.block8 = DownsampleBlock(512, 512)
        self.block9 = DLastConvBlock()

        self.from_rgb_512 = FromRGB(out_channels=512)
        self.from_rgb_256 = FromRGB(out_channels=256)
        self.from_rgb_128 = FromRGB(out_channels=128)
        self.from_rgb_64 = FromRGB(out_channels=64)
        self.from_rgb_32 = FromRGB(out_channels=32)
        self.from_rgb_16 = FromRGB(out_channels=16)

        _init_weights(self)

    def add_minibatch_std(self, x):
        b, _, h, w = x.shape
        # "We compute the standard deviation for each feature in each spatial location over the minibatch.
        # We then average these estimates over all features and spatial locations to arrive at a single value.
        # We replicate the value and concatenate it to all spatial locations and over the minibatch,
        # yielding one additional (constant) feature map."
        feat_map = x.std(dim=0, keepdim=True).mean(dim=(1, 2, 3), keepdim=True)
        x = torch.cat([x, feat_map.repeat(b, 1, h, w)], dim=1)
        return x

    def _fade_in(self, x, block, alpha):
        # "'$0.5\times$' refer to halving the image resolution using nearest neighbor average pooling."
        def _half(x):
            return F.avg_pool2d(x, kernel_size=2)

        skip = x.clone()
        skip = eval(f"""self.from_rgb_{block.in_channels}""")(skip)
        skip = block(skip)

        x = _half(x)
        x = eval(f"""self.from_rgb_{skip.shape[1]}""")(x)
        return (1 - alpha) * x + alpha * skip

    def forward(self, x, resol, alpha):
        if resol >= 1024:
            if resol == 1024:
                # "The first Conv $1 \times 1$ layer of the discriminator similarly corresponds to 'fromRGB'."
                x = self._fade_in(x=x, block=self.block1, alpha=alpha)
            x = self.block2(x) # `(b, 64, 256, 256)`
        if resol >= 512:
            if resol == 512:
                x = self._fade_in(x=x, block=self.block2, alpha=alpha)
            x = self.block3(x) # `(b, 128, 128, 128)`
        if resol >= 256:
            if resol == 256:
                x = self._fade_in(x=x, block=self.block3, alpha=alpha)
            x = self.block4(x) # `(b, 256, 64, 64)`
        if resol >= 128:
            if resol == 128:
                x = self._fade_in(x=x, block=self.block4, alpha=alpha)
            x = self.block5(x) # `(b, 512, 32, 32)`
        if resol >= 64:
            if resol == 64:
                x = self._fade_in(x=x, block=self.block5, alpha=alpha)
            x = self.block6(x) # `(b, 512, 16, 16)`
        if resol >= 32:
            if resol == 32:
                x = self._fade_in(x=x, block=self.block6, alpha=alpha)
            x = self.block7(x) # `(b, 512, 8, 8)`
        if resol >= 16:
            if resol == 16:
                x = self._fade_in(x=x, block=self.block7, alpha=alpha)
            x = self.block8(x) # `(b, 512, 4, 4)`
        if resol == 8:
            x = self._fade_in(x=x, block=self.block8, alpha=alpha)
        if resol == 4:
            x = self.from_rgb_512(x)

        # "We inject the across-minibatch standard deviation as an additional feature map
        # at $4 \times 4$ resolution toward the end of the discriminator."
        x = self.add_minibatch_std(x) # `(b, 513, 4, 4)`
        x = self.block9(x) # `(b, 1, 1, 1)`
        return x


if __name__ == "__main__":
    BATCH_SIZE = 1
    gen = Generator()
    x = torch.randn(BATCH_SIZE, 512, 1, 1)
    gen(x, resol=4, alpha=0.7).shape

    disc = Discriminator()
    alpha = 0.5
    resol = 4
    x = torch.randn((1, 3, resol, resol))
    disc(x, resol=resol, alpha=alpha).shape

# "EQUALIZED LEARNING RATE: W use a trivial $\mathcal{N}(0, 1)$ initialization and then
# explicitly scale the weights at runtime. We set $w^{^}_{i} = w_{i} / c$, where $w_{i}$ are the weights
# and $c$ is the per-layer normalization constant from He’s initializer."

# "GANs are prone to the escalation of signal magnitudes as a result of unhealthy competition
# between the two networks. We believe that the actual need in GANs is constraining signal magnitudes
# and competition. We use a different approach that consists of two ingredients, neither of which
# include learnable parameters."