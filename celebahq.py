# Source: https://www.kaggle.com/datasets/lamsimon/celebahq

# "We represent training and generated images in $[-1, 1]$".

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
# import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
# import random

from utils import get_image_dataset_mean_and_std


class CelebAHQDataset(Dataset):
    def __init__(self, data_dir, split="train", resol=1024):
        super().__init__()

        self.img_paths = list((Path(data_dir)/split).glob("**/*.jpg"))
        self.split = split
        self.resol = resol

        self.transformer = T.Compose([
            T.Resize(resol),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            # get_image_dataset_mean_and_std(data_dir)
            T.Normalize(mean=(0.517, 0.416, 0.363), std=(0.303, 0.275, 0.269)),
        ])

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        image = self.transformer(image)
        return image

    def __len__(self):
        return len(self.img_paths)


if __name__ == "__main__":
    data_dir = "/Users/jongbeomkim/Documents/datasets/celebahq/"
    ds = CelebAHQDataset(data_dir=data_dir)
    ds[10]