# References:
    # https://github.com/koshian2/swd-pytorch/blob/master/swd.py

import numpy as np
import torch
import torch.nn.functional as F


# Gaussian blur kernel
def get_gaussian_kernel(device="cpu"):
    kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]], np.float32) / 256.0
    gaussian_k = torch.as_tensor(kernel.reshape(1, 1, 5, 5)).to(device)
    return gaussian_k


def pyramid_down(image, device="cpu"):
    gaussian_k = get_gaussian_kernel(device=device)        
    # channel-wise conv(important)
    multiband = [F.conv2d(image[:, i:i + 1,:,:], gaussian_k, padding=2, stride=2) for i in range(3)]
    down_image = torch.cat(multiband, dim=1)
    return down_image


def pyramid_up(image, device="cpu"):
    gaussian_k = get_gaussian_kernel(device=device)
    upsample = F.interpolate(image, scale_factor=2)
    multiband = [F.conv2d(upsample[:, i:i + 1,:,:], gaussian_k, padding=2) for i in range(3)]
    up_image = torch.cat(multiband, dim=1)
    return up_image


def gaussian_pyramid(original, n_pyrams, device="cpu"):
    x = original
    # pyramid down
    pyrams = [original]
    for i in range(n_pyrams):
        x = pyramid_down(x, device=device)
        pyrams.append(x)
    return pyrams


def laplacian_pyramid(original, n_pyrams, device="cpu"):
    # create gaussian pyramid
    pyrams = gaussian_pyramid(original, n_pyrams, device=device)

    # pyramid up - diff
    laplacian = list()
    for i in range(len(pyrams) - 1):
        diff = pyrams[i] - pyramid_up(pyrams[i + 1], device=device)
        laplacian.append(diff)
    # Add last gaussian pyramid
    laplacian.append(pyrams[len(pyrams) - 1])        
    return laplacian


def minibatch_laplacian_pyramid(image, n_pyrams, batch_size, device="cpu"):
    b, _, _, _ = image.shape
    n = b // batch_size + np.sign(b % batch_size)
    pyrams = list()
    for i in range(n):
        x = image[i * batch_size:(i + 1) * batch_size]
        p = laplacian_pyramid(x.to(device), n_pyrams, device=device)
        p = [x.cpu() for x in p]
        pyrams.append(p)
    del x
    result = list()
    for i in range(n_pyrams + 1):
        x = list()
        for j in range(n):
            x.append(pyrams[j][i])
        result.append(torch.cat(x, dim=0))
    return result


def extract_patches(pyram_layer, slice_indices, slice_size=7, unfold_batch_size=128):
    assert pyram_layer.ndim == 4
    b, c, _, _ = pyram_layer.shape

    n = b // unfold_batch_size + np.sign(b % unfold_batch_size)
    # random slice 7x7
    p_slice = list()
    for i in range(n):
        # [unfold_batch_size, ch, n_slices, slice_size, slice_size]
        ind_start = i * unfold_batch_size
        ind_end = min((i + 1) * unfold_batch_size, b)
        x = pyram_layer[ind_start:ind_end].unfold(
            2, slice_size, 1).unfold(3, slice_size, 1).reshape(
            ind_end - ind_start, c, -1, slice_size, slice_size
        )
        # [unfold_batch_size, ch, n_descriptors, slice_size, slice_size]
        x = x[:, :, slice_indices, :, :]
        # [unfold_batch_size, n_descriptors, ch, slice_size, slice_size]
        p_slice.append(x.permute([0, 2, 1, 3, 4]))
    # sliced tensor per layer [batch, n_descriptors, ch, slice_size, slice_size]
    x = torch.cat(p_slice, dim=0)
    # normalize along ch
    std, mean = torch.std_mean(x, dim=(0, 1, 3, 4), keepdim=True)
    x = (x - mean) / (std + 1e-8)
    # reshape to 2rank
    x = x.reshape(-1, 3 * slice_size * slice_size)
    return x


@torch.no_grad()
def get_swd(
    image1,
    image2, 
    n_pyrams=None,
    slice_size=7,
    n_descriptors=128,
    n_repeat_proj=128,
    proj_per_repeat=4,
    return_by_resolution=False,
    pyram_batch_size=128,
    device="cpu",
):
    """
    `image1`, `image2`: Square size is recommended.
    `n_pyrams` : (Optional) Number of laplacian pyramids. If `None` (Same as in the paper), downsample pyramids
        toward 16×16 resolution. Output number of pyramids is `n_pyram + 1`,
        because lowest resolution gaussian pyramid is added to laplacian pyramids sequence.
    `slice_size`: (Optional) Patch size when slicing each layer of pyramids. Default is `7`
        (same as in the paper).
    `n_descriptors`: (Optional) Number of descriptors per image. Default is `128` (same as in the paper).
    `n_repeat_proje`: (Optional) Number of times to calculate a random projection. Please specify
        this value according to your GPU memory. Default is `128`. `n_repeat_proj * proj_per_repeat == 512`
        is recommended. This product value `512` is same as in the paper, but official implementation uses 4
        for `n_repeat_proj` and `128` for `proj_per_repeat`. (This method needs huge amount of memory...)
    `proj_per_repeat`: (Optional) Number of dimension to calculate a random projection on each repeat.
        Default is `4`. Higher value needs much more GPU memory. `n_repeat_proj * proj_per_repeat == 512`
        is recommended.
    `return_by_resolution`: (Optional) If `True`, returns SWD by each resolutions (laplacian pyramids).
        If `False`, returns the average of SWD values ​​by resolution. Default is `False`.
    `pyram_batch_size`: (Optional) Mini batch size of calculating laplacian pyramids.
        Higher value may cause CUDA out of memory error. This value does not affect on SWD estimation.
        Default is `128`.
    """
    assert image1.shape == image2.shape

    _, _, h, _ = image1.shape
    if n_pyrams is None:
        n_pyrams = int(np.rint(np.log2(h // 16)))

    # Minibatch laplacian pyramid for cuda memory reasons
    pyram1 = minibatch_laplacian_pyramid(
        image1, n_pyrams=n_pyrams, batch_size=pyram_batch_size, device=device,
    )
    pyram2 = minibatch_laplacian_pyramid(
        image2, n_pyrams=n_pyrams, batch_size=pyram_batch_size, device=device,
    )

    result = list()
    for idx in range(n_pyrams + 1):
        _, _, pyram_h, pyram_w = pyram1[idx].shape
        n = (pyram_h - 6) * (pyram_w - 6)
        indices = torch.randperm(n)[: n_descriptors]

        # Extract patches on CPU
        # patch : 2rank (n_image*n_descriptors, slice_size**2*ch)
        p1 = extract_patches(
            pyram_layer=pyram1[idx],
            slice_indices=indices,
            slice_size=slice_size,
        ).to(device)
        p2 = extract_patches(
            pyram_layer=pyram2[idx],
            slice_indices=indices,
            slice_size=slice_size,
        ).to(device)

        dists = list()
        for _ in range(n_repeat_proj):
            rand = torch.randn(p1.size(1), proj_per_repeat).to(device)  # (slice_size ** 2 * ch)
            rand = rand / torch.std(rand, dim=0, keepdim=True)  # Noramlize

            proj1 = torch.matmul(p1, rand)
            proj2 = torch.matmul(p2, rand)
            proj1, _ = torch.sort(proj1, dim=0)
            proj2, _ = torch.sort(proj2, dim=0)
            d = torch.abs(proj1 - proj2)
            d = d.mean()

            dists.append(d)

        result.append(torch.mean(torch.stack(dists)))
    
    # Average over resolution
    result = torch.stack(result) * 1e3
    if return_by_resolution:
        return result.cpu()
    else:
        return torch.mean(result).cpu()

if __name__ == "__main__":
    torch.manual_seed(123)

    batch_size = 40
    image1 = torch.rand(batch_size, 3, 64, 64)
    image2 = torch.rand(batch_size, 3, 64, 64)
    out = get_swd(image1, image2, n_repeat_proj=4, device="cpu")
