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


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_model(model):
    for p in model.parameters():
        p.requires_grad = True
