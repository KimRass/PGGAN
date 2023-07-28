from PIL import Image
from pathlib import Path
import numpy as np
from time import time
from datetime import timedelta
import torch
from torchvision.utils import make_grid
import numpy as np
from tqdm.auto import tqdm
import torchvision.transforms as T


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def save_image(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _to_pil(img).save(str(path))


def resize_by_repeating_pixels(img, resol):
    img = np.repeat(np.repeat(img, repeats=1024 // resol, axis=0), repeats=1024 // resol, axis=1)
    return img


def show_image(img):
    copied_img = img.copy()
    copied_img = _to_pil(copied_img)
    copied_img.show()


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"""Using {torch.cuda.device_count()} GPUs.""")
        # if torch.cuda.device_count() > 1:
        #     print(f"""Using {torch.cuda.device_count()} GPUs.""")
        # else:
        #     print("Using GPU.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device


def save_parameters(model, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(save_path))


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


def batched_image_to_grid(image, n_cols, mean=(0, 0, 0), std=(1, 1, 1)):
    b, _, h, w = image.shape
    assert b % n_cols == 0,\
        "The batch size should be a multiple of `n_cols` argument"
    pad = max(1, int(max(h, w) * 0.02))
    grid = make_grid(tensor=image, nrow=n_cols, normalize=False, padding=pad)
    grid = grid.clone().permute((1, 2, 0)).detach().cpu().numpy()

    grid *= std
    grid += mean
    grid *= 255.0
    grid = np.clip(a=grid, a_min=0, a_max=255).astype("uint8")

    for k in range(n_cols + 1):
        grid[:, (pad + h) * k: (pad + h) * k + pad, :] = 255
    for k in range(b // n_cols + 1):
        grid[(pad + h) * k: (pad + h) * k + pad, :, :] = 255
    return grid


def print_number_of_parameters(model):
    print(f"""{sum([p.numel() for p in model.parameters()]):,}""")


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))
