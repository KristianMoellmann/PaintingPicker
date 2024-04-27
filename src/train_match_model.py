import torch
import torch.nn as nn
import clip
import os
import json
import matplotlib.pyplot as plt
import random
import numpy as np
from custom_data import EmbeddedMatchDataset, EmbeddedMatchDataSplit, MatchDataHistorySplit
from torch.utils.data import DataLoader, random_split
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

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


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer, loss_func: callable, epochs: int, device: str, scoring: str, cv_index: int):

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
            torch.save(model.state_dict(), f'models/{scoring}/{os.path.basename(args.folder)}/{args.name}_cv{cv_index}.pt')

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        pbar.set_postfix({'Train loss': train_loss, 'Train acc': train_acc, 'Val loss': val_loss, 'Val acc': val_acc})      

    return train_losses, train_accs, val_losses, val_accs


# LOGIC REIMPLEMENTATION HERE
def initialize_dictionarioary(match_data):
    # Initialize the dictionary for tracking wins and losses
    init_wins_and_loses = {}     

    # Go trough each session which actually happend
    for session in match_data.keys():
        for match in match_data[session]:
            left_image = match_data[session][match]["left_image"]
            right_image = match_data[session][match]["right_image"]
            if left_image not in init_wins_and_loses:
                init_wins_and_loses[left_image] = {"W": set(), "L": set()}
            if right_image not in init_wins_and_loses:
                init_wins_and_loses[right_image] = {"W": set(), "L": set()}
            
            if match_data[session][match]["winner"] == 0:
                winner = match_data[session][match]["left_image"]
                loser = match_data[session][match]["right_image"]
                
            elif match_data[session][match]["winner"] == 1:
                loser = match_data[session][match]["left_image"]
                winner = match_data[session][match]["right_image"]
            
            else:
                draw_1 = match_data[session][match]["left_image"]
                draw_2 = match_data[session][match]["right_image"]
                continue
                # TODO implement draw
            
            init_wins_and_loses[winner]["W"].add(loser)
            init_wins_and_loses[loser]["L"].add(winner)
        
    return init_wins_and_loses

def find_transitative_wins_and_loses(transitative_wins_and_loses):
    pure_wins = dict()
    
    for image in transitative_wins_and_loses.keys():
        
        # Go trough search list for wins untill it is empty
        image_wins_set = set()
        win_search_list = list(transitative_wins_and_loses[image]["W"])
        exsisting_wins = set(win_search_list)
        while win_search_list:
            beaten_image = win_search_list.pop()
            if beaten_image not in image_wins_set:
                if beaten_image not in exsisting_wins:
                    image_wins_set.add(beaten_image)

                for beaten_image_win in transitative_wins_and_loses[beaten_image]["W"]:
                    win_search_list.append(beaten_image_win)
        
        # print(f"will beat {len(image_wins_set)} images: {image_wins_set}")  
            
        # Go trough search list for losses untill it is empty
        image_loss_set = set()
        loss_search_list = list(transitative_wins_and_loses[image]["L"])
        while loss_search_list:
            beaten_by_image = loss_search_list.pop()
            if beaten_by_image not in image_loss_set:
                image_loss_set.add(beaten_by_image)

                for beaten_image_loss in transitative_wins_and_loses[beaten_by_image]["L"]:
                    loss_search_list.append(beaten_image_loss) 
        
        # print(f"beaten by {len(image_loss_set)} images: {image_loss_set}")
        pure_wins[image] = image_wins_set - image_loss_set
    
        # pure wins due to transitative properties
        # print(f"pure wins {len(pure_wins[image])} images: {pure_wins[image]}")
    return pure_wins

def update_history(pure_wins, history):
    session = len(history.keys())
    history[session] = {}
    count = 0
    for image in pure_wins.keys():
        for loser in pure_wins[image]:
            count += 1
            if random.random() > 0.5:
                history[session][str(count)] = {"left_image": image, "right_image": loser, "winner": 0}
            else:
                history[session][str(count)] = {"left_image": loser, "right_image": image, "winner": 1}
    print(f"Added {count} matches to the history")
    return history

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help="Name of the user")
    parser.add_argument('--folder', default='data/processed/full', type=str, help="Folder containing images")
    parser.add_argument('--model', default='ViT-B/32', type=str, help="Model to use")
    parser.add_argument('--epochs', default=100, type=int, help="Number of epochs to train the model")
    parser.add_argument('--batch-size', default=32, type=int, help="Batch size to use")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate to use")
    parser.add_argument('--seed', default=500, type=int, help="Seed to use for reproducibility")
    parser.add_argument('--logic', action='store_true', help='Use the logic model to simulate training data')
    args = parser.parse_args()

    scoring = "elo"

    # Load the feautre extractor and the preprocess function
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args.folder = args.folder.replace('processed', 'embedded')
    # model = MatchInvariantNet().to(device)

    # Load the data
    scoring_path = Path(f'scores/{os.path.basename(args.folder)}/{scoring}/{args.name}_history.json')

    if not scoring_path.exists():
        raise FileNotFoundError(f"Scoring file {scoring_path} does not exist")
    
    with open(scoring_path, 'r') as f:
        labels = json.load(f)

    # Set seed
    torch.manual_seed(args.seed)

    # Initialise things which must be defined according to the scoring method
    data = None
    loss_func = None
    optimizer = None

    history_data = MatchDataHistorySplit(labels)
    _, history_test = history_data.hold_out(ratio=0.1, seed=args.seed)

    test_data_split = EmbeddedMatchDataSplit(args.folder, history_test, seed=args.seed)
    test_dataset = EmbeddedMatchDataset(test_data_split.data, test_data_split.target)    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    history_train_folds, history_val_folds = history_data.cross_fold(n_splits=10, seed=args.seed+1)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for cv_index, (history_train, history_val) in enumerate(zip(history_train_folds, history_val_folds)):

        model = MatchNet().to(device)

        val_data_split = EmbeddedMatchDataSplit(args.folder, history_val, seed=args.seed+2)
        val_dataset = EmbeddedMatchDataset(val_data_split.data, val_data_split.target)

        if args.logic:
            # Update history_train using logic to simulate
            # Initialize the dictionary for tracking wins and losses
            init_wins_and_loses = initialize_dictionarioary(history_train)
            # Find the transitative wins and losses
            pure_wins = find_transitative_wins_and_loses(init_wins_and_loses)
            # Update the history with the new matches
            history_train = update_history(pure_wins, history_train)
        
        train_data_split = EmbeddedMatchDataSplit(args.folder, history_train, seed=args.seed+3)
        train_dataset = EmbeddedMatchDataset(train_data_split.data, train_data_split.target)


        # Print the sizes of the datasets
        # print(f"Train size: {len(train_dataset)}")
        # print(f"Val size: {len(val_dataset)}")
        # print(f"Test size: {len(test_dataset)}")
        
        # Create data loader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Define the loss function and the optimizer
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train the model
        model_save_path = f'models/{scoring}/{os.path.basename(args.folder)}'
        os.makedirs(model_save_path, exist_ok=True)

        train_loss, train_acc, val_losse, val_acc = train(model, train_loader, val_loader, optimizer, loss_func, args.epochs, device, scoring, cv_index)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_losse)
        val_accs.append(val_acc)
    

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Plot the losses
    label_made = False
    for train_loss, train_acc, val_loss, val_acc in zip(train_losses, train_accs, val_losses, val_accs):
        if not label_made:
            axes[0].plot(train_loss, label='Train loss', color='blue')
            axes[0].plot(val_loss, label='Val loss', color='orange')
            axes[1].plot(train_acc, label='Train acc', color='blue')
            axes[1].plot(val_acc, label='Val acc', color='orange')
            label_made = True
        else:
            axes[0].plot(train_loss, color='blue')
            axes[0].plot(val_loss, color='orange')
            axes[1].plot(train_acc, color='blue')
            axes[1].plot(val_acc, color='orange')

    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.show()


    test_losses = []
    test_accs = []
    for i in range(10):
        # Load the best model
        model.load_state_dict(torch.load(f'models/{scoring}/{os.path.basename(args.folder)}/{args.name}_cv{i}.pt'))

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
        
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        print(f"CV {i} Test loss: {test_loss}, Test acc: {test_acc}")
    
    test_losses = np.array(test_losses)
    test_accs = np.array(test_accs)

    print(f"Mean test loss: {test_losses.mean()}, Std test loss: {test_losses.std()}")
    print(f"Mean test acc: {test_accs.mean()}, Std test acc: {test_accs.std()}")