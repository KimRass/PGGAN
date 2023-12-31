# Source: https://www.kaggle.com/datasets/lamsimon/celebahq

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

import config
from utils import get_image_dataset_mean_and_std


class CelebAHQDataset(Dataset):
    def __init__(self, data_dir, split="train", img_size=1024):
        super().__init__()

        self.img_paths = list((Path(data_dir)/split).glob("**/*.jpg"))
        self.split = split
        self.img_size = img_size

        self.transformer = T.Compose([
            T.Resize(img_size),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            # "We represent training and generated images in $[-1, 1]$."
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            # get_image_dataset_mean_and_std(data_dir)
            # T.Normalize(mean=(0.517, 0.416, 0.363), std=(0.303, 0.275, 0.269)),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        image = self.transformer(image)
        return image


def get_dataloader(split, batch_size, img_size):
    ds = CelebAHQDataset(data_dir=config.DATA_DIR, split=split, img_size=img_size)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.N_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    return dl


if __name__ == "__main__":
    data_dir = "/Users/jongbeomkim/Documents/datasets/celebahq/"
    ds = CelebAHQDataset(data_dir=data_dir, img_size=16)
