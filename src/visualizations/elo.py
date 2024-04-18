import json
import os
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def plot_elo_scores(scores: dict):
    
    all_elos = [val["elo"] for _, val in scores.items()]
    print(f"Min ELO: {min(all_elos):.2f}\nMax ELO: {max(all_elos):.2f}\nAverage ELO: {sum(all_elos) / len(all_elos):.2f}\nUnique ELOs: {len(set(all_elos))}")
    
    # all_elos_with_noise = [elo + (15 * (np.random.rand()*2 - 1)) for elo in all_elos]
    
    # Plot histogram of elos
    bins = int(len(set(all_elos)) / 2)
    # figsize
    plt.figure(figsize=(5, 3.5))
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)  # Add grid lines behind histogram
    plt.hist(all_elos, bins=bins, zorder=3, label='Original')
    # plt.hist(all_elos_with_noise, bins=bins, zorder=4, alpha=0.4, color='red', label='Simulation')
    plt.xlabel("ELO")
    plt.ylabel("Frequency")
    plt.legend()
    # plt.title(f"ELO distribution\nMin: {min(all_elos):.2f}, Max: {max(all_elos):.2f}, Average: {sum(all_elos) / len(all_elos):.2f}, Unique: {len(set(all_elos))}")
    plt.tight_layout()
    plt.savefig(f"reports/figures/elo/elo_distribution_{args.name}.pdf")
    plt.show()


def plot_elo_history(history: dict):
    sessions_time = []
    sessions_votes = [0, 0, 0]
    for session_elos in history.values():
        session_times = []
        last_time = None
        for time, data in session_elos.items():
            current_time = datetime.fromisoformat(time)
            if last_time is not None:
                # Compute time difference
                delta = current_time - last_time
                if delta.total_seconds() < 100:
                    sessions_time.append(delta.total_seconds())
                else:
                    print(f"Time difference of {delta.total_seconds()}s is too large")
            last_time = current_time
            sessions_votes[data["winner"]] += 1

    # Plot histogram of time differences
    plt.hist(sessions_time, bins=100)
    plt.xlabel("Time difference between votes (s)")
    plt.ylabel("Frequency")
    plt.title("Time difference between votes")
    # Add average time difference with label
    plt.axvline(x=sum(sessions_time) / len(sessions_time), color='r', linestyle='dashed', linewidth=1)
    plt.text(sum(sessions_time) / len(sessions_time), 200, f"Average: {sum(sessions_time) / len(sessions_time):.2f}", rotation=-90)
    plt.tight_layout()
    plt.savefig(f"reports/figures/elo/time_difference_{args.name}.pdf")
    plt.show()

    # Plot bar chart of votes with counts on top of the bars
    plt.bar(range(len(sessions_votes)), sessions_votes)
    for i, count in enumerate(sessions_votes):
        plt.text(i, count, str(count), ha='center', va='bottom')
    plt.xticks(range(len(sessions_votes)), ["Left", "Right", "Tie"])
    plt.title("Votes")
    plt.tight_layout()
    plt.savefig(f"reports/figures/elo/votes_{args.name}.pdf")
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help="Name of the user")
    parser.add_argument('--folder', default='data/processed/full', type=str, help="Folder containing images")
    parser.add_argument('--plot', default='both', choices=['both', 'scores', 'history'], type=str, help="Plot to show (scores or history)")
    args = parser.parse_args()

    scores_file = Path(f'scores/{os.path.basename(args.folder)}/elo/{args.name}.json')
    history_file = Path(f'scores/{os.path.basename(args.folder)}/elo/{args.name}_history.json')

    if not scores_file.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_file}")

    if not history_file.exists():
        raise FileNotFoundError(f"History file not found: {history_file}")
    
    if not os.path.exists('reports/figures/elo'):
        os.makedirs('reports/figures/elo')
    
    with open(scores_file, 'r') as f:
        scores = json.load(f)

    if args.plot == 'scores' or args.plot == 'both':
        plot_elo_scores(scores)
    
    with open(history_file, 'r') as f:
        history = json.load(f)

    if args.plot == 'history' or args.plot == 'both':
        plot_elo_history(history)