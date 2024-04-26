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
from PIL import Image



def get_predictions_elo(model: nn.Module, test_loader: DataLoader, loss_func: callable, device: str, r_min=800, r_max=1800):
    # Plot predictions on test set
    test_predictions = []
    test_names = []
    test_targets = []
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for image, target, name in test_loader:
            image = image.to(device)
            pred = model(image)
            test_predictions.append(pred)
            test_names.append(name)
            test_targets.append(target)
            debug = 1

    test_predictions = torch.round(torch.cat(test_predictions).cpu()) * ((r_max - r_min) + r_min)
    test_targets = torch.cat(test_targets).cpu()
    new_test_names = []
    for name in test_names:
        new_name = str(name).split("[")[1].split("]")[0]
        new_test_names.append(f"{new_name}.jpg")
        
    # Plot histogram of test predictions and targets
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = ['skyblue', 'salmon']
    titles = ['Predictions', 'Targets']
    for ax, data, title, color in zip(axes, [test_predictions, test_targets], titles, colors):
        ax.hist(data, bins=50, alpha=0.75, label=title, color=color)
        ax.axvline(data.mean(), color='darkred', linestyle='dashed', linewidth=2, label='Mean')
        ax.set_xlim(r_min-5, r_max+5)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)  # Adding grid lines

    fig.suptitle(f"Predictions on test set, test loss: {test_loss:.2f}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust the top to accommodate the suptitle
    plt.show()
    
    return test_predictions, test_targets, new_test_names


def get_9(predictions, names):
    # Get the 9 best, 9 worst and 9 middle rated images
    sorted_predictions = predictions.argsort()
    
    paired_sorted = sorted(zip(predictions, names))

    # Unzip back into two lists
    sorted_predictions, sorted_names = zip(*paired_sorted)
    
    # we get help from Anna, Elisabeth, Mikkel, Vi
    best_9_predictions = sorted_predictions[-9:]
    best_9_names = sorted_names[-9:]
    worst_9_predictions = sorted_predictions[:9]
    worst_9_names = sorted_names[:9]
    middle_9_predictions = sorted_predictions[len(sorted_predictions)//2-4:len(sorted_predictions)//2+5]
    middle_9_names = sorted_names[len(sorted_predictions)//2-4:len(sorted_predictions)//2+5]
    
    return best_9_predictions, best_9_names, worst_9_predictions, worst_9_names, middle_9_predictions, middle_9_names


def plot_3x3grid(pictures, labels, title: str):
    # Prepares full paths and converts labels to integers if they are in tensor format
    paths = [f'data/processed/unseen/{name}' for name in pictures]
    labels = [label.item() if hasattr(label, 'item') else label for label in labels]
    
    # Setup the figure and axes
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))  # Larger figure size
    axes = axes.flatten()  # Flatten the grid to make indexing easier

    for i, (path, label) in enumerate(zip(paths, labels)):
        img = Image.open(path)  # Load image from path
        axes[i].imshow(img)  # Display image
        axes[i].set_title(f'Label: {round(label)}', fontsize=12, fontweight='bold', color='blue')  # Enhanced title styling
        axes[i].axis('off')  # Turn off the axis

        # Optionally, you can add a border around each image
        for spine in axes[i].spines.values():  # Adding a frame with color to the image
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

    plt.suptitle(title, fontsize=16, fontweight='bold', color='red')  # Enhanced main title styling
    plt.tight_layout(pad=3.0)  # Increase padding between plots
    plt.subplots_adjust(top=0.92)  # Adjust top margin to make space for the main title
    plt.show()  # Display the plot
    

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help="Name of the user")
    # parser.add_argument('--folder', default='data/processed/unseen', type=str, help="Folder containing images")
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size to use")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate to use")
    args = parser.parse_args()

    # Load the feautre extractor and the preprocess function
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # args.folder = args.folder.replace('processed', 'embedded')
    model = ScaleNetV2().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()

    # Load the best model
    model.load_state_dict(torch.load(f'models/elo/full/{args.name}.pt'))
    
    # We need to get the r_min, r_max for normalising the predictions
    train_scoring_path = Path(f'scores/full/elo/{args.name}.json')
    with open(train_scoring_path, 'r') as f:
        train_labels = json.load(f)
        
    train_data = EmbeddedEloDataset("data/embedded/full", train_labels)
    r_min, r_max = train_data.r_min, train_data.r_max
    
    test_scoring_path = Path(f'scores/unseen/elo/{args.name}.json')
    with open(test_scoring_path, 'r') as f:
        test_labels = json.load(f)
    
    test_data = EmbeddedEloDataset('data/embedded/unseen', test_labels)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False) 
    
    predictions, test_targets, test_names = get_predictions_elo(model, test_loader, loss_func, device, r_min, r_max)
    
    best_9_preds, best_9_names, worst_9_preds, worst_9_names, middle_9_preds, middle_9_names = get_9(predictions, test_names)
    
    plot_3x3grid(worst_9_names, worst_9_preds, "worst")
    plot_3x3grid(middle_9_names, middle_9_preds, "middle")
    plot_3x3grid(best_9_names, best_9_preds, "best")
    
    debug = 1 