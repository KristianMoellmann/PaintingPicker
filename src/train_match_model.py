import torch
import torch.nn as nn
import clip
import os
import json
import matplotlib.pyplot as plt
from custom_data import EmbeddedMatchDataset, EmbeddedMatchDataSplit, MatchDataHistorySplit
from torch.utils.data import DataLoader, random_split
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

class MatchNet(nn.Module):

    def __init__(self, input_size: int=512, p: float=0.5) -> None:
        super(MatchNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(2*input_size, 128),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(16, 2),
        )
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (B, 2, 512)
        # Stack the two embeddings in the second dimenson
        x1 = torch.hstack([x[:, 0, :], x[:, 1, :]])
        # Stack the opposite way
        x2 = torch.hstack([x[:, 1, :], x[:, 0, :]])
        x1 = self.seq(x1)
        x2 = self.seq(x2)

        # x1 and x2 have shape (B, 2)
        # Add the two outputs but change the order of the second one
        x = x1 + torch.cat([x2[:, 1].unsqueeze(1), x2[:, 0].unsqueeze(1)], dim=1)
        # OTHER IDEA: MULTIPLY THE SECOND OUTPUT BY -1
        # x = x1 - x2
        return x


class MatchInvariantNet(nn.Module):

    def __init__(self, input_size: int=512, p: float=0.5) -> None:
        super(MatchInvariantNet, self).__init__()
        self.l1 = nn.Linear(input_size, 128)  # Change 512 to the number of features extracted by the feature extractor
        self.r = nn.ReLU()
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 1)
        # self.l4 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has size (B, 2, 512)
        x = self.l1(x)
        x = self.r(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.r(x)
        x = self.dropout(x)
        x = self.l3(x)
        # x = self.r(x)
        # x = self.dropout(x)
        # x = self.l4(x)
        # x has size (B, 2, 1)
        x = x.squeeze(2)
        return x


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer, loss_func: callable, epochs: int, device: str, scoring: str):

    model.train()
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    best_val_loss = float('inf')

    for epoch in (pbar := tqdm(range(epochs))):
        model.train()
        pbar.set_description(f"Epoch {epoch+1}/{epochs}")
        train_loss = 0
        train_acc = 0
        for image, target in train_loader:
            image = image.to(device)
            target = target.to(device)

            pred = model(image)
            loss = loss_func(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()/len(train_loader)
            train_acc += (pred.argmax(1) == target).sum().item()/len(train_loader.dataset)

        with torch.no_grad():
            model.eval()
            val_acc = 0
            val_loss = 0
            for image, target in val_loader:
                image = image.to(device)
                target = target.to(device)

                pred = model(image)
                loss = loss_func(pred, target)

                val_loss += loss.item()/len(val_loader)
                val_acc += (pred.argmax(1) == target).sum().item()/len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'models/{scoring}/{os.path.basename(args.folder)}/{args.name}.pt')

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        pbar.set_postfix({'Train loss': train_loss, 'Train acc': train_acc, 'Val loss': val_loss, 'Val acc': val_acc})      

    return train_losses, train_accs, val_losses, val_accs

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help="Name of the user")
    parser.add_argument('--folder', default='data/processed/full', type=str, help="Folder containing images")
    parser.add_argument('--model', default='ViT-B/32', type=str, help="Model to use")
    parser.add_argument('--epochs', default=100, type=int, help="Number of epochs to train the model")
    parser.add_argument('--batch-size', default=32, type=int, help="Batch size to use")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate to use")
    parser.add_argument('--seed', default=42, type=int, help="Seed to use for reproducibility")
    parser.add_argument('--logic', action='store_true', help='Use the logic model to simulate training data')
    args = parser.parse_args()

    scoring = "elo"

    # Load the feautre extractor and the preprocess function
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args.folder = args.folder.replace('processed', 'embedded')
    model = MatchNet().to(device)
    # model = MatchInvariantNet().to(device)

    # Load the data
    scoring_path = Path(f'scores/{os.path.basename(args.folder)}/{scoring}/{args.name}_history.json')

    if not scoring_path.exists():
        raise FileNotFoundError(f"Scoring file {scoring_path} does not exist")
    
    with open(scoring_path, 'r') as f:
        labels = json.load(f)

    # Initialise things which must be defined according to the scoring method
    data = None
    loss_func = None
    optimizer = None

    history_data = MatchDataHistorySplit(labels, seed=args.seed)
    history_train_val, history_test = history_data.hold_out_test(ratio=0.1)

    test_data = EmbeddedMatchDataSplit(args.folder, history_test)

    if not args.logic:
        data = EmbeddedMatchDataSplit(args.folder, labels, seed=args.seed)

    test_data, test_targets = data.hold_out_test(ratio=0.1)

    (train_data, train_targets), (val_data, val_targets) = data.split_data(ratio=0.8, seed=args.seed+1)

    # Print the sizes of the datasets
    print(f"Train size: {len(train_data)}")
    print(f"Val size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")

    train_dataset = EmbeddedMatchDataset(train_data, train_targets)
    val_dataset = EmbeddedMatchDataset(val_data, val_targets)
    test_dataset = EmbeddedMatchDataset(test_data, test_targets)
    
    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Define the loss function and the optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    model_save_path = f'models/{scoring}/{os.path.basename(args.folder)}'
    os.makedirs(model_save_path, exist_ok=True)

    train_losses, train_accs, val_losses, val_accs = train(model, train_loader, val_loader, optimizer, loss_func, args.epochs, device, scoring)

    # Plot the losses
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation losses')
    plt.show()

    # Plot the accuracies
    plt.plot(train_accs, label='Training accuracy')
    plt.plot(val_accs, label='Validation accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and validation accuracies')
    plt.show()

    # Load the best model
    model.load_state_dict(torch.load(f'models/{scoring}/{os.path.basename(args.folder)}/{args.name}.pt'))

    # Plot predictions on test set
    test_predictions = []
    test_targets = []
    test_loss = 0
    test_acc = 0
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
            test_acc += (pred.argmax(1) == target).sum().item()/len(test_loader.dataset)
    
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_acc}")