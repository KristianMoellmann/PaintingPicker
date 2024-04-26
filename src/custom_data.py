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
        self.data, self.target, self.names = self.load_images(image_folder, labels)

    def load_images(self, image_folder: str, labels: dict):
        data = []
        target = []
        names = []
        for image_name, scoring in labels.items():
            if image_name == '2023-04-05 16.09.44.jpg':
                names.append("1010101010.jpg")
            else:
                names.append(image_name)
            image_name = image_name.replace('.jpg', '.pt')
            image_file = os.path.join(image_folder, image_name)
            data.append(torch.load(image_file))
            score = scoring['elo']
            target.append(score)

        data = torch.stack(data)
        target = torch.tensor(target, dtype=torch.float32)
        names = [int(name.split('.')[0]) for name in names]
        names = torch.tensor(names, dtype=torch.int)

        # Normalize the target between 0 and 1
        self.r_min = target.min().item()
        self.r_max = target.max().item()
        if self.r_max - self.r_min != 0:
            target = (target - self.r_min) / (self.r_max - self.r_min)

        return data, target, names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.target[idx]
        name = self.names[idx]    
        return sample, target, name


class MatchDataHistorySplit:

    def __ini__(self, history: dict, seed: int = None):
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
        for session, history in self.history.items():
            new_history[session] = {}
            for time, scoring in history.items():
                if scoring["left_image"] in images or scoring["right_image"] in images:
                    new_history[session][time] = scoring
                    self.history[session].pop(time)
                
        return new_history
    
    def hold_out_test(self, ratio: float = 0.2):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        n = len(self.remaining_images)
        indices = torch.randperm(n)
        test_index = int(n * ratio)

        test_images = self.remaining_images[indices[:test_index]]
        self.remaining_images = self.remaining_images[indices[test_index:]]

        test_history = self.create_new_history(test_images)
        
        return self.history, test_history



class EmbeddedMatchDataSplit:

    def __init__(self, image_folder: str, labels: dict, seed: int = None):
        self.data, self.target, self.image_matches = self.load_images(image_folder, labels)
        self.seed = seed
    
    def load_images(self, image_folder: str, labels: dict):
        data = []
        target = []
        image_matches = defaultdict(lambda: [])
        count = 0
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
                image_matches[img_1].append((count, img_2))
                image_matches[img_2].append((count, img_1))
                count += 1
        data = torch.stack(data)
        target = torch.tensor(target, dtype=torch.int64)

        return data, target, image_matches
    
    def hold_out_test(self, ratio: float = 0.2):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        images = np.array(list(self.image_matches.keys()))
        n = len(images)
        indices = torch.randperm(n)
        test_index = int(n * ratio)
        test_images = images[indices[:test_index]]
        test_data_indices = []
        test_target_indices = []
        for image in test_images:
            for idx, opponent in self.image_matches[image]:
                test_data_indices.append(idx)
                test_target_indices.append(idx)

                # Remove the match from the image_matches of the other image
                self.image_matches[opponent].remove((idx, image))

            del self.image_matches[image]
        
        test_data = self.data[test_data_indices]
        test_targets = self.target[test_target_indices]

        return (test_data, test_targets)
    

    def split_data(self, ratio: float = 0.8, seed: int = None):
        if seed is not None:
            torch.manual_seed(seed)
        
        # Validation
        images = np.array(list(self.image_matches.keys()))
        n = len(images)
        indices = torch.randperm(n)
        val_index = int(n * (1 - ratio))
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