import json
import os
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


def plot_elo_scores(scores: dict):
    pass


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
    plt.show()

    # Plot bar chart of votes with counts on top of the bars
    plt.bar(range(len(sessions_votes)), sessions_votes)
    for i, count in enumerate(sessions_votes):
        plt.text(i, count, str(count), ha='center', va='bottom')
    plt.xticks(range(len(sessions_votes)), ["Left", "Right", "Tie"])
    plt.title("Votes")
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help="Name of the user")
    parser.add_argument('--folder', default='data/fewer_imgs', type=str, help="Folder containing images")
    args = parser.parse_args()

    scores_file = Path(f'scores/{os.path.basename(args.folder)}/elo/{args.name}.json')
    history_file = Path(f'scores/{os.path.basename(args.folder)}/elo/{args.name}_history.json')

    if not scores_file.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_file}")

    if not history_file.exists():
        raise FileNotFoundError(f"History file not found: {history_file}")
    
    with open(scores_file, 'r') as f:
        scores = json.load(f)

    plot_elo_scores(scores)
    
    with open(history_file, 'r') as f:
        history = json.load(f)

    plot_elo_history(history)