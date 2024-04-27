import torch
import torch.nn as nn
import numpy as np
import json
import os
import random
from glob import glob
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime

# Model
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


def create_random_image_pairings(image_names: list):
    # Pair each image with another image randomly. If odd number, one image will have two pairings
    image_pairings = []
    names = image_names.copy()
    while len(names) > 1:
        image1 = names.pop(random.randint(0, len(names)-1))
        image2 = names.pop(random.randint(0, len(names)-1))
        image_pairings.append((image1, image2))
    if len(names) == 1:
        random_image = random.choice(image_names)
        while random_image == names[0]:
            random_image = random.choice(image_names)
        image_pairings.append((names[0], random_image))
    return image_pairings


def calculate_elo(ra, rb, sa, sb, K=32):
    """
    Calculate the new Elo ratings for two players.

    Parameters:
    ra (float): The current rating of player A.
    rb (float): The current rating of player B.
    sa (float): The score of player A (1 for win, 0.5 for draw, 0 for loss).
    sb (float): The score of player B (1 for win, 0.5 for draw, 0 for loss).
    K (int, optional): The K-factor, which determines how much the ratings change. Default is 32.

    Returns:
    tuple: The new ratings for player A and player B.
    """
        
    # Calculate the expected score for each player
    Ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    Eb = 1 / (1 + 10 ** ((ra - rb) / 400))
    
    # Update the ratings
    ra_new = ra + K * (sa - Ea)
    rb_new = rb + K * (sb - Eb)
    
    return ra_new, rb_new


def update_elo_scores(ratings, image1, image2, prediction):
    # Get the current ratings
    image1_rating = ratings[image1]["elo"]
    image2_rating = ratings[image2]["elo"]

    # Update the ratings
    if prediction == 0:
        image1_rating, image2_rating = calculate_elo(image1_rating, image2_rating, 1, 0)
        ratings[image1]["wins"] += 1
        ratings[image2]["losses"] += 1
    elif prediction == 1:
        image1_rating, image2_rating = calculate_elo(image1_rating, image2_rating, 0, 1)
        ratings[image1]["losses"] += 1
        ratings[image2]["wins"] += 1

    # Update the ratings dictionary
    ratings[image1]["elo"] = image1_rating
    ratings[image1]["matches"] += 1

    ratings[image2]["elo"] = image2_rating
    ratings[image2]["matches"] += 1

    return ratings


def update_elo_history(history, image1, image2, prediction, session_index):
    history[session_index][datetime.now().isoformat()] = {
        "left_image": image1,
        "right_image": image2,
        "winner": prediction
    }
    return history



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("--dataset", type=str, default="full")
    parser.add_argument("--num-matches", type=int, default=10_000)
    args = parser.parse_args()

    # Load current ratings
    ratings_path = Path(f"scores/{args.dataset}/elo/{args.name}.json")
    with open(ratings_path, "r") as f:
        ratings = json.load(f)
    
    # Load history
    history_path = Path(f"scores/{args.dataset}/elo/{args.name}_history.json")
    with open(history_path, "r") as f:
        history = json.load(f)
    session_index = len(history)
    history[session_index] = {}        

    # Load the model
    model = MatchNet()
    model.load_state_dict(torch.load(f"models/{args.dataset}/elo/match/{args.name}.pt"))
    model.eval()

    # Get all image names
    images = glob(f"data/processed/{args.dataset}/*.jpg")
    image_names = [os.path.basename(image) for image in images]

    match_count = 0
    done = False

    pbar = tqdm(total=args.num_matches)
    while not done:
        image_pairings = create_random_image_pairings(image_names)
        
        for image1, image2 in image_pairings:
            # Get the embeddings
            image1_embedding = torch.load(f"data/embedded/{args.dataset}/{image1.replace('.jpg', '.pt')}")
            image2_embedding = torch.load(f"data/embedded/{args.dataset}/{image2.replace('.jpg', '.pt')}")
            embeddings = torch.vstack([image1_embedding, image2_embedding]).unsqueeze(0)

            # Get the prediction
            prediction = torch.argmax(model(embeddings)).item()
            

            # Update the ratings
            ratings = update_elo_scores(ratings, image1, image2, prediction)
            history = update_elo_history(history, image1, image2, prediction, session_index)

            match_count += 1
            pbar.update(1)
            pbar.set_postfix({"Match count": match_count})

            if match_count >= args.num_matches:
                done = True
                break
    
    # Save the updated ratings
    new_elo_path = Path(f"scores/{args.dataset}/elo/{args.name}_clip.json")
    with open(new_elo_path, "w") as f:
        json.dump(ratings, f, indent=4)

    # Save the updated history
    new_history_path = Path(f"scores/{args.dataset}/elo/{args.name}_clip_history.json")
    with open(new_history_path, "w") as f:
        json.dump(history, f, indent=4)
    