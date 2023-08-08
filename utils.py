import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from PIL import Image
from pathlib import Path
from time import time
from datetime import timedelta
from tqdm.auto import tqdm

import config


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def save_image(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _to_pil(img).save(str(path))


def show_image(img):
    copied_img = img.copy()
    copied_img = _to_pil(copied_img)
    copied_img.show()


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
    # ckpt["D"] = disc.module.state_dict()
    # ckpt["G"] = gen.module.state_dict()
    if config.N_GPUS > 1 and config.MULTI_GPU:
        ckpt["D"] = disc.module.state_dict()
        ckpt["G"] = gen.module.state_dict()
    else:
        ckpt["D"] = disc.state_dict()
        ckpt["G"] = gen.state_dict()
    torch.save(ckpt, str(save_path))


def get_image_dataset_mean_and_std(data_dir, ext="jpg"):
    data_dir = Path(data_dir)

    sum_rgb = 0
    sum_rgb_square = 0
    sum_resol = 0
    for img_path in tqdm(list(data_dir.glob(f"""**/*.{ext}"""))):
        pil_img = Image.open(img_path)
        tensor = T.ToTensor()(pil_img)
        
        sum_rgb += tensor.sum(dim=(1, 2))
        sum_rgb_square += (tensor ** 2).sum(dim=(1, 2))
        _, h, w = tensor.shape
        sum_resol += h * w
    mean = torch.round(sum_rgb / sum_resol, decimals=3)
    std = torch.round((sum_rgb_square / sum_resol - mean ** 2) ** 0.5, decimals=3)
    return mean, std


def image_to_grid(image, n_cols, value_range=(-1, 1)):
    _, _, h, w = image.shape
    image = image.repeat_interleave(1024 // h, dim=2).repeat_interleave(1024 // w, dim=3)
    grid = make_grid(image, nrow=n_cols, padding=20, normalize=True, value_range=value_range, pad_value=1)
    grid = TF.to_pil_image(grid)
    return grid


def print_number_of_parameters(model):
    print(f"""{sum([p.numel() for p in model.parameters()]):,}""")


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))


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


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_model(model):
    for p in model.parameters():
        p.requires_grad = True
