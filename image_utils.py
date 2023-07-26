from PIL import Image
from pathlib import Path
import numpy as np


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
