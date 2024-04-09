import matplotlib.pyplot as plt
import numpy as np
import json
import os
from argparse import ArgumentParser
from pathlib import Path


def plot_scale_count(scores: dict, scale: int):
    """
    Plot the count of each scale in the scores dictionary

    Args:
        scores (dict): The scores dictionary {"img_id": score}
    """
    scales = np.arange(1, scale + 1)
    counts = np.zeros(scale)
    for score in scores.values():
        counts[score - 1] += 1

    plt.figure(figsize=(10, 5))
    plt.bar(scales, counts)
    plt.xlabel('Scale')
    plt.ylabel('Count')
    plt.title('Count of each scale')
    plt.tight_layout()
    plt.show()

def plot_history(scores: dict, scale: int):
    """
    Plot the history of the scores

    Args:
        scores (dict): The scores dictionary {"img_id": score}
    """
    scales = np.arange(1, scale + 1)
    counts = np.zeros(scale)
    history = []
    for score in scores.values():
        counts[score - 1] += 1
        history.append(counts.copy())

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    for i in range(scale):
        axes[0].plot(np.arange(1, len(history) + 1), [h[i] for h in history], label=f'Scale {i + 1}')
    axes[0].set_xlabel('Image')
    axes[0].set_ylabel('Count')
    axes[0].set_title('History of each scale')
    axes[0].legend()

    axes[1].plot(np.arange(1, len(history) + 1), scores.values())
    axes[1].set_xlabel('Image')
    axes[1].set_ylabel('Scale')
    axes[1].set_title('Scale per image')
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help="Name of the user")
    parser.add_argument('--folder', default='data/fewer_imgs', type=str, help="Folder containing images")
    parser.add_argument('--scale', type=int, default=9, help="Number of scales")
    args = parser.parse_args()

    scores_file = Path(f'scores/{os.path.basename(args.folder)}/scale_{args.scale}/{args.name}.json')

    if not scores_file.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_file}")
    
    with open(scores_file, 'r') as f:
        scores = json.load(f)
    
    plot_scale_count(scores, args.scale)
    plot_history(scores, args.scale)