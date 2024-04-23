import torch
import torch.nn as nn

import json
import matplotlib.pyplot as plt
from custom_data import EmbeddedMatchDataset, EmbeddedMatchDataSplit
from torch.utils.data import DataLoader, random_split
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help="Name of the user")
    parser.add_argument('--dataset', default='full', type=str, help="Folder containing images")
    parser.add_argument('--model', default='ViT-B/32', type=str, help="Model to use")
    parser.add_argument('--seed', default=42, type=int, help="Seed to use for reproducibility")
    parser.add_argument('--split', action='store_true', help="Whether to split the data")
    args = parser.parse_args()

    scoring_path = Path(f'scores/{args.folder}/elo/{args.name}_history.json')

    if not scoring_path.exists():
        raise FileNotFoundError(f"Scoring file {scoring_path} does not exist")
    
    with open(scoring_path, 'r') as f:
        labels = json.load(f)

