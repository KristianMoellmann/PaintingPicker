import torch
import torch.nn as nn
import clip
import os
import json
import matplotlib.pyplot as plt
from custom_data import ScaleDataset, EmbeddedScaleDataset, EmbeddedEloDataset
from torch.utils.data import DataLoader, random_split
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
# from src.models.model import ScaleNet
from train_model import ScaleNetV2


def get_predictions_elo(model: nn.Module, test_loader: DataLoader, loss_func: callable, device: str):
    # Plot predictions on test set
    test_predictions = []
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for image, target in test_loader:
            image = image.to(device)

            pred = model(image)
            test_predictions.append(pred)

    test_predictions = torch.round(torch.cat(test_predictions).cpu())
    
    return test_predictions


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help="Name of the user")
    parser.add_argument('--folder', default='data/processed/unseen', type=str, help="Folder containing images")
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size to use")
    args = parser.parse_args()

    # Load the feautre extractor and the preprocess function
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.embed_now:
        clip_model, preprocess = clip.load(args.model, device=device)

        # Create the model
        feature_extractor = clip_model.visual
        for param in feature_extractor.parameters():
            param.requires_grad = False
        feature_extractor.float()
        feature_extractor.eval()

        model = ScaleNetWithClip(feature_extractor).to(device)
    
    else:
        args.folder = args.folder.replace('processed', 'embedded')
        model = ScaleNetV2().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()

    
    # TODO Now use the best model to predict the ratings of 1000 unseen images.
    # Then look at the 9 worst rated, the 9 in the middle and the 9 best rated images.
    
    # Load the best model
    model.load_state_dict(torch.load(f'models/elo/full/{args.name}.pt'))
    
    scoring_path = Path(f'scores/unseen/elo/{args.name}.json')
    with open(scoring_path, 'r') as f:
        labels = json.load(f)
    
    
    test_data = EmbeddedScaleDataset('data/processed/unseen', None)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False) 
    
    get_predictions_elo(model, test_loader, loss_func, device)
        
        
        