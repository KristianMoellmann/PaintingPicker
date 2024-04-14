import torch
import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose
import einops
from PIL import Image

class ScaleDataset(Dataset):

    def __init__(self, image_folder: str, labels: dict, preprocess: Compose, scale: int = 9):
        self.data, self.target = self.load_images(image_folder, labels, scale)
        self.preprocess = preprocess
        self.scale = scale

    def load_images(self, image_folder: str, labels: dict, scale: int):
        data = []
        target = []
        for image_name, scoring in labels.items():
            image_file = os.path.join(image_folder, image_name)
            data.append(Image.open(image_file))
            target.append(scoring)
        target = torch.tensor(target, dtype=torch.float32)

        # Normalize the target between 0 and 1
        target = (target - 1) / (scale - 1)

        return data, target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.preprocess(self.data[idx])
        target = self.target[idx]
            
        return sample, target