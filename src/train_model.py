import torch
import torch.nn as nn
import clip
import os
import json
from custom_data import ScaleDataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
# from src.models.model import ScaleNet


class ScaleNet(nn.Module):

    def __init__(self, feature_extractor: nn.Module) -> None:
        super(ScaleNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
        self.l1 = nn.Linear(feature_extractor.output_dim, 256)  # Change 512 to the number of features extracted by the feature extractor
        self.r = nn.ReLU()
        self.l2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        x = self.l1(features)
        x = self.r(x)
        x = self.l2(x)
        x = self.sigmoid(x)
        return x.flatten()


def train(model: nn.Module, dataloader: DataLoader, optimizer, loss_fn, epochs: int, device: str, scoring: str):

    model.train()

    for epoch in (pbar := tqdm(range(epochs))):
        pbar.set_description(f"Epoch {epoch+1}/{epochs}")
        for image, target in dataloader:
            image = image.to(device)
            target = target.to(device)

            pred = model(image)
            loss = loss_func(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'Loss': loss.item()})

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help="Name of the user")
    parser.add_argument('--folder', default='data/processed/full', type=str, help="Folder containing images")
    parser.add_argument('--scoring', default='scale_9', type=str, choices=['elo', 'scale_9'], help="Scoring method to use")
    parser.add_argument('--model', default='ViT-B/32', type=str, help="Model to use")
    parser.add_argument('--epochs', default=10, type=int, help="Number of epochs to train the model")
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size to use")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate to use")
    args = parser.parse_args()

    # Load the feautre extractor and the preprocess function
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(args.model, device=device)

    # Create the model
    feature_extractor = clip_model.visual
    for param in feature_extractor.parameters():
        param.requires_grad = False
    feature_extractor.eval()

    model = ScaleNet(feature_extractor).to(device)

    # Load the data
    scoring_path = Path(f'scores/{os.path.basename(args.folder)}/{args.scoring}/{args.name}.json')

    if not scoring_path.exists():
        raise FileNotFoundError(f"Scoring file {scoring_path} does not exist")
    
    with open(scoring_path, 'r') as f:
        labels = json.load(f)

    # Initialise things which must be defined according to the scoring method
    data = None
    loss_func = None
    optimizer = None

    if args.scoring == 'elo':
        raise NotImplementedError("Elo scoring is not implemented yet")
    else:
        data = ScaleDataset(args.folder, labels, preprocess)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Create data loader
    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    train(model, data_loader, optimizer, loss_func, args.epochs, device, args.scoring)