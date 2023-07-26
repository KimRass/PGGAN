# Source: https://www.kaggle.com/datasets/lamsimon/celebahq

# "We represent training and generated images in $[-1, 1]$".

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
import random

from utils import get_image_dataset_mean_and_std


class CelebAHQDataset(Dataset):
    def __init__(self, root, split="train", resol=1024):
        super().__init__()

        self.img_paths = list((Path(root)/split).glob("**/*.jpg"))
        self.split = split
        self.resol = resol

        self.transformer = T.Compose([
            # "When training the discriminator, we feed in real images that are downscaled to match
            # the current resolution of the network."
            T.Resize(resol),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            # get_image_dataset_mean_and_std(root)
            T.Normalize(mean=(0.517, 0.416, 0.363), std=(0.303, 0.275, 0.269)),
        ])

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        image = self.transformer(image)
        return image

    def __len__(self):
        return len(self.img_paths)


if __name__ == "__main__":
    root = "/Users/jongbeomkim/Documents/datasets/celebahq/"
    ds = CelebAHQDataset(root=root)
    ds[10]