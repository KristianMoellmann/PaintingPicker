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


class ScaleNetWithClip(nn.Module):

    def __init__(self, feature_extractor: nn.Module) -> None:
        super(ScaleNetWithClip, self).__init__()
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


class ScaleNet(nn.Module):
    name: str = 'ScaleNet'

    def __init__(self, input_size: int=512) -> None:
        super(ScaleNet, self).__init__()
        self.l1 = nn.Linear(input_size, 10)
        self.r = nn.ReLU()
        self.l2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.r(x)
        x = self.l2(x)
        x = self.sigmoid(x)
        return x.flatten()


class ScaleNetV2(nn.Module):
    name: str = 'ScaleNetV2'

    def __init__(self, input_size: int=512) -> None:
        super(ScaleNetV2, self).__init__()
        self.l1 = nn.Linear(input_size, 128)
        self.r = nn.ReLU()
        self.dropout = nn.Dropout(0.7)  # Add dropout layer
        self.l2 = nn.Linear(128, 32)  # Add another linear layer
        self.l3 = nn.Linear(32, 1)  # Add final linear layer
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.r(x)
        x = self.dropout(x)  # Apply dropout
        x = self.l2(x)
        x = self.r(x)
        x = self.dropout(x)  # Apply dropout
        x = self.l3(x)
        x = self.sigmoid(x)
        return x.flatten()


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer, loss_func: callable, epochs: int, device: str, scoring: str):

    model.train()
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')

    for epoch in (pbar := tqdm(range(epochs))):
        pbar.set_description(f"Epoch {epoch+1}/{epochs}")
        train_loss = 0
        for image, target in train_loader:
            image = image.to(device)
            target = target.to(device)

            pred = model(image)
            loss = loss_func(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()/len(train_loader)

        with torch.no_grad():
            val_loss = 0
            for image, target in val_loader:
                image = image.to(device)
                target = target.to(device)

                pred = model(image)
                loss = loss_func(pred, target)

                val_loss += loss.item()/len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'models/{scoring}/{os.path.basename(args.folder)}/{args.name}.pt')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        pbar.set_postfix({'Train loss': train_loss, 'Val loss': val_loss})      

    print(f"Best validation loss: {best_val_loss:.3f}")

    return train_losses, val_losses


def plot_predictions_scale9(model: nn.Module, test_loader: DataLoader, loss_func: callable, device: str):
    # Plot predictions on test set
    test_predictions = []
    test_targets = []
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for image, target in test_loader:
            image = image.to(device)
            target = target.to(device)

            pred = model(image)
            test_predictions.append(pred)
            test_targets.append(target)
            loss = loss_func(pred, target)
            test_loss += loss.item()/len(test_loader)
    
    test_predictions = torch.round(torch.cat(test_predictions).cpu() * 8 + 1)
    test_targets = torch.cat(test_targets).cpu() * 8 + 1

    # Plot histofram of test predicitons
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))

    for i, ax in enumerate(axes.flatten()):
        # Bar plot of predictions
        ax.hist(test_predictions[test_targets == i+1], bins=9, alpha=0.5, label='Predictions', color='blue')
        # plot mean of predictions
        ax.axvline(test_predictions[test_targets == i+1].mean(), color='red', linestyle='dashed', linewidth=2)
        ax.set_xlim(1, 10)
        # ax.set_ylim(0, 30)
        ax.xaxis.set_ticks(range(0, 11))
        ax.set_title(f"True rating: {i+1}")
    fig.suptitle(f"Predictions on test set, test loss: {test_loss:.2f}")
    plt.tight_layout()
    plt.show()


def plot_predictions_elo(model: nn.Module, test_loader: DataLoader, loss_func: callable, device: str):
    # Plot predictions on test set
    test_predictions = []
    test_targets = []
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for image, target in test_loader:
            image = image.to(device)
            target = target.to(device)

            pred = model(image)
            test_predictions.append(pred)
            test_targets.append(target)
            loss = loss_func(pred, target)
            test_loss += loss.item()/len(test_loader)

    test_predictions = torch.round(torch.cat(test_predictions).cpu() * (r_max - r_min) + r_min)
    test_targets = torch.cat(test_targets).cpu() * (r_max - r_min) + r_min

    # Plot histofram of test predicitons and targets
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, data, title in zip(axes, [test_predictions, test_targets], ['Predictions', 'Targets']):
        ax.hist(data, bins=100, alpha=0.5, label=title, color='blue')
        ax.axvline(data.mean(), color='red', linestyle='dashed', linewidth=2)
        ax.set_xlim(r_min-5, r_max+5)
        # ax.set_ylim(0, 30)
        # ax.xaxis.set_ticks(range(int(r_min), int(r_max)+1))
        ax.set_title(title)
    fig.suptitle(f"Predictions on test set, test loss: {test_loss:.2f}")
    plt.tight_layout()
    plt.savefig(f"reports/figures/training/{args.name}_{args.scoring}_{args.score_type}_{model.name}_predictions.pdf")
    plt.show()


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help="Name of the user")
    parser.add_argument('--folder', default='data/processed/full', type=str, help="Folder containing images")
    parser.add_argument('--scoring', default='scale_9', type=str, choices=['elo', 'scale_9'], help="Scoring method to use")
    parser.add_argument('--model', default='ViT-B/32', type=str, help="Model to use")
    parser.add_argument('--epochs', default=100, type=int, help="Number of epochs to train the model")
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size to use")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate to use")
    parser.add_argument('--embed_now', action='store_true', help="Embed the images now")
    parser.add_argument('--seed', default=42, type=int, help="Seed to use for reproducibility")
    parser.add_argument('--score_type', default='original', choices=['original', 'logic', 'clip'], help="Decide which score type to use")
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

    # Load the data
    if args.score_type == 'original':
        scoring_path = Path(f'scores/{os.path.basename(args.folder)}/{args.scoring}/{args.name}.json')
    elif args.score_type == 'logic':
        scoring_path = Path(f'scores/{os.path.basename(args.folder)}/{args.scoring}/{args.name}_logic.json')
    elif args.score_type == 'clip':
        scoring_path = Path(f'scores/{os.path.basename(args.folder)}/{args.scoring}/{args.name}_clip.json')
    else:
        raise ValueError("score_type must be one of 'original', 'logic', 'clip'")

    if not scoring_path.exists():
        raise FileNotFoundError(f"Scoring file {scoring_path} does not exist")
    
    with open(scoring_path, 'r') as f:
        labels = json.load(f)

    # Initialise things which must be defined according to the scoring method
    data = None
    loss_func = None
    optimizer = None

    if args.scoring == 'elo':
        if args.embed_now:
            raise ValueError("Elo scoring is not supported with 'embed_now' option. Please embed the images first using src/data/embed_dataset.py.")
        else:
            data = EmbeddedEloDataset(args.folder, labels)
            r_min, r_max = data.r_min, data.r_max
    else:
        if args.embed_now:
            preprocess = clip_model.transform
            data = ScaleDataset(args.folder, labels, preprocess)
        else:
            data = EmbeddedScaleDataset(args.folder, labels)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Split the data
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    n = len(data)

    train_data, val_data, test_data = random_split(data, [int(train_ratio*n), int(val_ratio*n), int(test_ratio*n)])
    
    # Create data loader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Train the model
    model_save_path = f'models/{args.scoring}/{os.path.basename(args.folder)}'
    os.makedirs(model_save_path, exist_ok=True)

    train_losses, val_losses = train(model, train_loader, val_loader, optimizer, loss_func, args.epochs, device, args.scoring)

    # Plot the losses
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and validation losses. Best val loss: {min(val_losses):.3f}')
    plt.savefig(f"reports/figures/training/{args.name}_{args.scoring}_{args.score_type}_{model.name}_loss.pdf")
    plt.show()

    # Load the best model
    model.load_state_dict(torch.load(f'models/{args.scoring}/{os.path.basename(args.folder)}/{args.name}.pt'))

    if args.scoring == 'scale_9':
        plot_predictions_scale9(model, test_loader, loss_func, device)
    else:
        plot_predictions_elo(model, test_loader, loss_func, device)
    