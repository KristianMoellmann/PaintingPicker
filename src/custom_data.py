import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose
import einops
from PIL import Image
from typing import Tuple
from collections import defaultdict


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


class MatchDataHistorySplit:

    def __init__(self, history: dict):
        self.history = history
        self.remaining_images = self.get_images_from_history()

    def get_images_from_history(self):
        images = set()
        for session, history in self.history.items():
            for time, scoring in history.items():
                images.add(scoring["left_image"])
                images.add(scoring["right_image"])
        return np.array(list(images))

    def create_new_history(self, images: np.array):
        new_history = {}
        remaining_history = {}
        for session, history in self.history.items():
            new_history[session] = {}
            remaining_history[session] = {}
            for time, scoring in history.items():
                if scoring["left_image"] in images or scoring["right_image"] in images:
                    new_history[session][time] = scoring
                else:
                    remaining_history[session][time] = scoring
        return new_history, remaining_history
    
    def hold_out(self, ratio: float = 0.2, seed: int = None):
        if seed is not None:
            torch.manual_seed(seed)
        
        n = len(self.remaining_images)
        indices = torch.randperm(n)
        test_index = int(n * ratio)

        test_images = self.remaining_images[indices[:test_index]]
        self.remaining_images = self.remaining_images[indices[test_index:]]

        test_history, self.history = self.create_new_history(test_images)
        
        return self.history, test_history

    def cross_fold(self, n_splits: int = 10, seed: int = None):
        if seed is not None:
            torch.manual_seed(seed)
        
        n = len(self.remaining_images)
        indices = torch.randperm(n)
        fold_size = n // n_splits

        val_folds = []
        train_folds = []
        for i in range(n_splits):
            val_index = indices[i*fold_size:(i+1)*fold_size]
            val_images = self.remaining_images[val_index]

            val_history, train_history = self.create_new_history(val_images)
            val_folds.append(val_history)
            train_folds.append(train_history)
        
        return train_folds, val_folds


class EmbeddedMatchDataSplit:

    def __init__(self, image_folder: str, labels: dict, seed: int = None):
        self.seed = seed
        self.data, self.target, self.image_matches = self.load_images(image_folder, labels)        
    
    def load_images(self, image_folder: str, labels: dict):
        data = []
        target = []
        image_matches = defaultdict(lambda: [])
        count = 0
        np.random.seed(self.seed)
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
                if np.random.rand() > 0.5:
                    data.append(torch.cat([embed_1, embed_2]))
                    target.append(scoring["winner"])
                else:
                    data.append(torch.cat([embed_2, embed_1]))
                    target.append(1 - scoring["winner"])
                image_matches[img_1].append((count, img_2))
                image_matches[img_2].append((count, img_1))
                count += 1
        data = torch.stack(data)
        target = torch.tensor(target, dtype=torch.int64)

        return data, target, image_matches
    
    def split_data(self, train_atio: float = 0.8, seed: int = None):
        if seed is not None:
            torch.manual_seed(seed)
        
        # Validation
        images = np.array(list(self.image_matches.keys()))
        n = len(images)
        indices = torch.randperm(n)
        val_index = int(n * (1 - train_atio))
        val_images = images[indices[:val_index]]
        val_data_indices = []
        val_target_indices = []

        for image in val_images:
            for idx, opponent in self.image_matches[image]:
                val_data_indices.append(idx)
                val_target_indices.append(idx)

                # Remove the match from the image_matches of the other image
                self.image_matches[opponent].remove((idx, image))

            del self.image_matches[image]
        
        val_data = self.data[val_data_indices]
        val_targets = self.target[val_target_indices]

        # Train
        train_images = images[indices[val_index:]]
        train_data_indices = []
        train_target_indices = []

        for image in train_images:
            for idx, opponent in self.image_matches[image]:
                train_data_indices.append(idx)
                train_target_indices.append(idx)

        train_data = self.data[train_data_indices]
        train_targets = self.target[train_target_indices]
        
        return (train_data, train_targets), (val_data, val_targets)


class EmbeddedMatchDataset(Dataset):

    def __init__(self, data: torch.Tensor, target: torch.Tensor, create_duplicates: bool=False):
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