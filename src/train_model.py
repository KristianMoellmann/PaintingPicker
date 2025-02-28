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
from sklearn.model_selection import KFold
# from src.models.model import ScaleNet
from scipy import stats
import yaml


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
    
class ScaleNetV3(nn.Module):
    name: str = 'ScaleNetV3'

    def __init__(self, input_size: int=512) -> None:
        super(ScaleNetV3, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.seq(x)
        return out.flatten()

class ScaleNetV4(nn.Module):
    name: str = 'ScaleNetV4'

    def __init__(self, input_size: int=512) -> None:
        super(ScaleNetV4, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.seq(x)
        return out.flatten()

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer, loss_func: callable, epochs: int, device: str, scoring: str):

    model.train()
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model = None

    # pbar = tqdm(range(epochs), leave=True)
    for epoch in range(epochs):
        # pbar.set_description(f"Epoch {epoch+1}/{epochs}")
        train_loss = 0
        for image, target, name in train_loader:  # remove 'name' if args.scoring == 'scale_9'
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
            for image, target, name in val_loader:  # remove 'name' if args.scoring == 'scale_9'
                image = image.to(device)
                target = target.to(device)

                pred = model(image)
                loss = loss_func(pred, target)

                val_loss += loss.item()/len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # pbar.set_postfix({'Train loss': train_loss, 'Val loss': val_loss})      

    # print(f"Best validation loss: {best_val_loss:.3f}")

    return train_losses, val_losses, best_model

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
    tau = stats.kendalltau(test_targets, test_predictions).statistic

    if not args.dont_plot:
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

        # Calculate Kendall's tau

        # Scatter plot of test predicitons and targets
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(test_targets, test_predictions, alpha=0.5)
        ax.plot([0, 10], [0, 10], color='red', linestyle='dashed', linewidth=2)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_xlabel('True rating')
        ax.set_ylabel('Predicted rating')
        ax.set_title(f"Predictions on test set, test loss: {test_loss:.2f}, Kendall's tau corr.: {tau:.3f}")
        plt.tight_layout()
        plt.savefig(f"reports/figures/training/{args.name}_{args.scoring}_{args.score_type}_{model.name}_scatter.pdf")
        plt.show()
    # print(f"Test loss: {test_loss:.3f}")

    return test_loss, tau

def plot_predictions_elo(model: nn.Module, test_loader: DataLoader, loss_func: callable, device: str):
    # Plot predictions on test set
    test_predictions = []
    test_targets = []
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for image, target, name in test_loader:
            image = image.to(device)
            target = target.to(device)

            pred = model(image)
            test_predictions.append(pred)
            test_targets.append(target)
            loss = loss_func(pred, target)
            test_loss += loss.item()/len(test_loader)
    
    # print(f"Test loss: {test_loss:.3f}")
    test_predictions = torch.round(torch.cat(test_predictions).cpu() * (r_max - r_min) + r_min)
    test_targets = torch.cat(test_targets).cpu() * (r_max - r_min) + r_min
    tau = stats.kendalltau(test_targets, test_predictions).statistic

    if not args.dont_plot:
        # Scatter plot of test predicitons and targets
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(test_targets, test_predictions, alpha=0.5)
        ax.plot([r_min-10, r_max+10], [r_min-10, r_max+10], color='red', linestyle='dashed', linewidth=2)
        ax.set_xlim(r_min-10, r_max+10)
        ax.set_ylim(r_min-10, r_max+10)
        ax.set_xlabel('True rating')
        ax.set_ylabel('Predicted rating')
        ax.set_title(f"Predictions on test set, test loss: {test_loss:.2f}, Kendall's tau corr.: {tau:.3f}")
        plt.tight_layout()
        plt.savefig(f"reports/figures/training/{args.name}_{args.scoring}_{args.score_type}_{model.name}_scatter.pdf")
        plt.show()

    return test_loss, tau


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
    parser.add_argument('--dont_plot', action='store_true', help="Plot the predictions on the test set")
    parser.add_argument('--regressor', default='ScaleNetV2', choices=['ScaleNet', 'ScaleNetV2', 'ScaleNetV3', 'ScaleNetV4'], help="Regressor to use")
    parser.add_argument('--save_results', action='store_true', help="Save the results to a yaml file")
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load the feautre extractor and the preprocess function
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    labels_train_val, labels_test = random_split(labels, [int(0.90*len(labels)), int(0.10*len(labels))])

    labels_test_key = [[*labels_test.dataset.keys()][i] for i in labels_test.indices]
    labels_test = {k: labels_test.dataset[k] for k in labels_test_key}

    kf = KFold(n_splits=9, shuffle=True, random_state=args.seed)
    test_losses = []
    test_kendalls_tau_corr = []
    
    kf_iterator = tqdm(kf.split(labels_train_val), total=kf.get_n_splits(), leave=False)
    # enumerate(kf.split(labels_train_val.indices))
    for train_index, val_index in kf_iterator:
        # print(f"\nFold {i+1}/9")
        train_index = [labels_train_val.indices[idx] for idx in train_index]
        labels_train_key = [[*labels_train_val.dataset.keys()][i] for i in train_index]
        labels_train = {k: labels_train_val.dataset[k] for k in labels_train_key}
        
        val_index = [labels_train_val.indices[idx] for idx in val_index]
        labels_val_key = [[*labels_train_val.dataset.keys()][i] for i in val_index]
        labels_val = {k: labels_train_val.dataset[k] for k in labels_val_key}

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
            if args.regressor == 'ScaleNet':
                model = ScaleNet().to(device)
            elif args.regressor == 'ScaleNetV2':
                model = ScaleNetV2().to(device)
            elif args.regressor == 'ScaleNetV3':
                model = ScaleNetV3().to(device)
            elif args.regressor == 'ScaleNetV4':
                model = ScaleNetV4().to(device)
        
        # Initialise things which must be defined according to the scoring method
        data = None
        loss_func = None
        optimizer = None

        if args.name == 'kristoffer':
            args.folder = args.folder.replace('full', 'full_kris')

        if args.scoring == 'elo':
            if args.embed_now:
                raise ValueError("Elo scoring is not supported with 'embed_now' option. Please embed the images first using src/data/embed_dataset.py.")
            else:
                train_data = EmbeddedEloDataset(args.folder, labels_train)
                val_data = EmbeddedEloDataset(args.folder, labels_val)
                test_data = EmbeddedEloDataset(args.folder, labels_test)
                r_min, r_max = train_data.r_min, train_data.r_max
        else:
            if args.embed_now:
                preprocess = clip_model.transform
                train_data = ScaleDataset(args.folder, labels_train, preprocess)
                val_data = ScaleDataset(args.folder, labels_val, preprocess)
                test_data = ScaleDataset(args.folder, labels_test, preprocess)
            else:
                train_data = EmbeddedScaleDataset(args.folder, labels_train)
                val_data = EmbeddedScaleDataset(args.folder, labels_val)
                test_data = EmbeddedScaleDataset(args.folder, labels_test)

        if args.name == 'kristoffer':
            args.folder = args.folder.replace('full_kris', 'full')  

        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # Create data loader
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

        # Train the model
        if args.scoring == 'scale_9':
            model_dir_path = f'models/{os.path.basename(args.folder)}/{args.scoring}'
        else:
            model_dir_path = f'models/{os.path.basename(args.folder)}/{args.scoring}/{args.score_type}'

        os.makedirs(model_dir_path, exist_ok=True)

        train_losses, val_losses, best_model = train(model, train_loader, val_loader, optimizer, loss_func, args.epochs, device, args.scoring)

        if args.scoring == 'scale_9':
            model_path = f'models/{os.path.basename(args.folder)}/{args.scoring}/{args.name}.pt'
        else:
            model_path = f'models/{os.path.basename(args.folder)}/{args.scoring}/{args.score_type}/{args.name}.pt'

        torch.save(best_model.state_dict(), model_path)
        

        if not args.dont_plot:
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
        model.load_state_dict(torch.load(model_path))

        if args.scoring == 'scale_9':
            test_loss, tau = plot_predictions_scale9(model, test_loader, loss_func, device)
        else:
            test_loss, tau = plot_predictions_elo(model, test_loader, loss_func, device)
        
        test_losses.append(test_loss)
        test_kendalls_tau_corr.append(tau)
        kf_iterator.set_postfix({'Test loss': test_loss, 'Kendall\'s tau corr': tau})
    
    avg_test_loss = sum(test_losses)/len(test_losses)
    avg_kendalls_tau_corr = sum(test_kendalls_tau_corr)/len(test_kendalls_tau_corr)
    if not args.save_results:
        print(f"\n{args.name} results:")
        print(f"    - Average test loss: {avg_test_loss:.3f}")
        print(f"    - Average Kendall's tau correlation: {avg_kendalls_tau_corr:.3f}")
    

    if args.save_results:
        # Save results to yaml file
        path = f'results/full/'
        path_to_file = os.path.join(path, f"{args.scoring}.yaml")

        if not os.path.exists(path):
            os.makedirs(path)

        # Check if the file exists
        if not os.path.isfile(path_to_file):
            results = {args.score_type: {args.regressor: {args.name: {}}}}
        else: # read the file and add new key if needed
            with open(path_to_file, 'r') as f:
                results = yaml.load(f, Loader=yaml.FullLoader)
            results.setdefault(args.score_type, {}).setdefault(args.regressor, {}).setdefault(args.name, {})

        results[args.score_type][args.regressor][args.name] = {
            'test_losses': [float(test_loss) for test_loss in test_losses], 
            'test_kendalls_tau_corr': [float(corr) for corr in test_kendalls_tau_corr], 
            'avg_test_loss': float(avg_test_loss), 
            'avg_kendalls_tau_corr': float(avg_kendalls_tau_corr)
        }

        with open(path_to_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=True)