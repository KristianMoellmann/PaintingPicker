import torch
import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose
import einops
from PIL import Image
from typing import Tuple


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


class EmbeddedScaleDataset(Dataset):

    def __init__(self, image_folder: str, labels: dict, scale: int = 9):
        self.data, self.target = self.load_images(image_folder, labels, scale)
        self.scale = scale

    def load_images(self, image_folder: str, labels: dict, scale: int):
        data = []
        target = []
        for image_name, scoring in labels.items():
            image_name = image_name.replace('.jpg', '.pt')
            image_file = os.path.join(image_folder, image_name)
            data.append(torch.load(image_file))
            target.append(scoring)

        data = torch.stack(data)
        target = torch.tensor(target, dtype=torch.float32)

        # Normalize the target between 0 and 1
        target = (target - 1) / (scale - 1)

        return data, target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.target[idx]
            
        return sample, target


class EmbeddedEloDataset(Dataset):

    def __init__(self, image_folder: str, labels: dict):
        self.data, self.target = self.load_images(image_folder, labels)

    def load_images(self, image_folder: str, labels: dict):
        data = []
        target = []
        for image_name, scoring in labels.items():
            image_name = image_name.replace('.jpg', '.pt')
            image_file = os.path.join(image_folder, image_name)
            data.append(torch.load(image_file))
            score = scoring['elo']
            target.append(score)

        data = torch.stack(data)
        target = torch.tensor(target, dtype=torch.float32)

        # Normalize the target between 0 and 1
        self.r_min = target.min().item()
        self.r_max = target.max().item()
        target = (target - self.r_min) / (self.r_max - self.r_min)

        return data, target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.target[idx]
            
        return sample, target


class EmbeddedMatchDataSplit:

    def __init__(self, image_folder: str, labels: dict, split: Tuple[float], seed: int = None):
        self.data, self.target = self.load_images(image_folder, labels)
        self.split = split
        self.seed = seed
    
    def load_images(self, image_folder: str, labels: dict):
        data = []
        target = []
        for session, history in labels.items():
            for time, scoring in history.items():
                # Skip ties
                if scoring["winner"] == 2:
                    continue
                img_1 = scoring["left_image"].replace('.jpg', '.pt')
                img_2 = scoring["right_image"].replace('.jpg', '.pt')
                image_file_1 = os.path.join(image_folder, img_1)
                image_file_2 = os.path.join(image_folder, img_2)
                embed_1 = torch.load(image_file_1)
                embed_2 = torch.load(image_file_2)
                data.append(torch.cat([embed_1, embed_2]))
                target.append(scoring["winner"])

        data = torch.stack(data)
        target = torch.tensor(target, dtype=torch.int64)

        return data, target
    

    def split_data(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        n = len(self.data)
        indices = torch.randperm(n)
        train_index = int(n * self.split[0])
        val_index = int(n * (self.split[0] + self.split[1]))

        train_data = self.data[indices[:train_index]]
        train_targets = self.target[indices[:train_index]]
        val_data = self.data[indices[train_index:val_index]]
        val_targets = self.target[indices[train_index:val_index]]
        test_data = self.data[indices[val_index:]]
        test_targets = self.target[indices[val_index:]]

        return (train_data, train_targets), (val_data, val_targets), (test_data, test_targets)


class EmbeddedMatchDataset(Dataset):

    def __init__(self, data: torch.Tensor, target: torch.Tensor, create_duplicates: bool=True):
        self.data, self.target = data, target
        self.create_duplicates = create_duplicates
        if self.create_duplicates:
            self.data, self.target = self.create_duplicate_data()
    
    def create_duplicate_data(self):
        # data has size (n, 2, 512)
        # swap the two images in each pair
        swapped_data = self.data[:, [1, 0], :]
        swapped_target = 1 - self.target
        data = torch.cat([self.data, swapped_data])
        target = torch.cat([self.target, swapped_target])
        return data, target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.target[idx]
            
        return sample, target