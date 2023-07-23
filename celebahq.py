# Source: https://www.kaggle.com/datasets/lamsimon/celebahq

# "We represent training and generated images in $[-1, 1]$".

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
import random

from torch_utils import get_image_dataset_mean_and_std


class CelebAHQDataset(Dataset):
    def __init__(self, root, split="train"):
        super().__init__()

        self.img_paths = list((Path(root)/split).glob("**/*.jpg"))

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        return image

    def __len__(self):
        return len(self.img_paths)
root = "/Users/jongbeomkim/Downloads/celeba_hq"
get_image_dataset_mean_and_std(root)
ds = CelebAHQDataset(root=root)
ds[10].show()