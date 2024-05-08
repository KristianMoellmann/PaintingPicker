import json
import os
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_elo_scores(score_dict: dict, normalized=False):

    # figsize
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)  # Add grid lines behind histogram
    COLORS = {
        "Original": "black",
        "Logic": "red",
        "Clip": "blue"
    }

    lns = []

    for name, scores in score_dict.items():
    
        all_elos = np.array([val["elo"] for _, val in scores.items()])
        print(f"Min ELO: {min(all_elos):.2f}\nMax ELO: {max(all_elos):.2f}\nAverage ELO: {sum(all_elos) / len(all_elos):.2f}\nUnique ELOs: {len(set(all_elos))}")
        
        if normalized:
            all_elos = (all_elos - all_elos.min()) / (all_elos.max() - all_elos.min())

        # all_elos_with_noise = [elo + (15 * (np.random.rand()*2 - 1)) for elo in all_elos]
        
        # Plot histogram of elos
        bins = int(len(set(all_elos)) / 10)
        if name == "Original":
            ln = ax.hist(all_elos, bins=20, zorder=5, label=name, color=COLORS[name], alpha=0.7, density=True)
        elif name == "Logic":
            ln = ax.hist(all_elos, bins=50, zorder=4, label=name, color=COLORS[name], alpha=0.5, density=True)
        else:
            ln = ax.hist(all_elos, bins=50, zorder=3, label=name, color=COLORS[name], alpha=0.5, density=True)
        
        lns.append(ln[2][0])
    
    ax.legend(lns, [l.get_label() for l in lns], loc='upper right')

    if normalized:
        ax.set_xlabel("Normalised ELO")
    else:
        ax.set_xlabel("ELO")
    # ylabel since it is a density
    ax.set_ylabel("Frequency")
    # plt.title(f"ELO distribution\nMin: {min(all_elos):.2f}, Max: {max(all_elos):.2f}, Average: {sum(all_elos) / len(all_elos):.2f}, Unique: {len(set(all_elos))}")
    plt.tight_layout()
    if normalized:
        plt.savefig(f"reports/figures/elo/elo_distribution_{args.name}_normalized.pdf")
    else:
        plt.savefig(f"reports/figures/elo/elo_distribution_{args.name}.pdf")
    plt.show()


def plot_elo_scores_scatter(score_dict: dict):

    # figsize
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)  # Add grid lines behind histogram
    COLORS = {
        "Original": "black",
        "Logic": "red",
        "Clip": "blue"
    }

    sorting = np.argsort(np.array([val["elo"] for _, val in score_dict["Original"].items()]))

    for name, scores in score_dict.items():
    
        all_elos = np.array([val["elo"] for _, val in scores.items()])[sorting]
        all_elos = (all_elos - all_elos.min()) / (all_elos.max() - all_elos.min())

        print(f"Min ELO: {min(all_elos):.2f}\nMax ELO: {max(all_elos):.2f}\nAverage ELO: {sum(all_elos) / len(all_elos):.2f}\nUnique ELOs: {len(set(all_elos))}")
        
        # all_elos_with_noise = [elo + (15 * (np.random.rand()*2 - 1)) for elo in all_elos]


        if name != "Original":
            # # For each bin calculate mean and standard deviation
            # bin_means, bin_edges, _ = stats.binned_statistic(range(len(all_elos)), all_elos, statistic='mean', bins=10)
            # bin_std, _, _ = stats.binned_statistic(range(len(all_elos)), all_elos, statistic='std', bins=10)

            # # Compute mean and std of the last half bin and add to means and stds
            # last_mean = np.mean(all_elos[int(np.mean(bin_edges[-2:])):])
            # last_std = np.std(all_elos[int(np.mean(bin_edges[-2:])):])
            # bin_means = np.append(bin_means, last_mean)
            # bin_std = np.append(bin_std, last_std)

            bin_width = 100
            bins = 20
            bin_centers = np.arange(0, len(all_elos)+1, len(all_elos)//bins)
            bin_means = np.zeros(len(bin_centers))
            bin_std = np.zeros(len(bin_centers))
            for i, center in enumerate(bin_centers):
                start = max(0, center - bin_width//2)
                end = min(len(all_elos), center + bin_width//2)
                bin_means[i] = np.mean(all_elos[start:end])
                bin_std[i] = np.std(all_elos[start:end])

            # Plot fill between the mean and standard deviation
            if name == "Logic":
                ax.fill_between(bin_centers, bin_means - bin_std*2, bin_means + bin_std*2, alpha=0.3, color=COLORS[name], zorder=4)
            else:
                ax.fill_between(bin_centers, bin_means - bin_std*2, bin_means + bin_std*2, alpha=0.3, color=COLORS[name], zorder=3)
        
        # Plot histogram of elos
        # ax.plot(range(len(all_elos)), all_elos, zorder=3, label=name, color=COLORS[name], alpha=0.5)
        if name == "Original":
            ax.scatter(range(len(all_elos)), all_elos, zorder=5, label=name, color=COLORS[name], s=2)
        elif name == "Logic":
            ax.scatter(range(len(all_elos)), all_elos, zorder=4, label=name, color=COLORS[name], s=2)
        else:
            ax.scatter(range(len(all_elos)), all_elos, zorder=3, label=name, color=COLORS[name], s=2)

    
    ax.legend(loc='upper left')    
    ax.set_xlabel("Index")
    ax.set_ylabel("Normalised ELO")
    ax.set_xlim(0, len(all_elos))
    # plt.title(f"ELO distribution\nMin: {min(all_elos):.2f}, Max: {max(all_elos):.2f}, Average: {sum(all_elos) / len(all_elos):.2f}, Unique: {len(set(all_elos))}")
    plt.tight_layout()
    plt.savefig(f"reports/figures/elo/elo_scatter_{args.name}.pdf")
    plt.show()


def plot_elo_scores_scatter_new(score_dict: dict):

    # figsize
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)  # Add grid lines behind histogram
    COLORS = {
        "Logic": "red",
        "Clip": "blue"
    }

    # sorting = np.argsort(np.array([val["elo"] for _, val in score_dict["Original"].items()]))

    original_elos = np.array([val["elo"] for _, val in score_dict["Original"].items()])
    sorting = np.argsort(original_elos)

    original_elos = ((original_elos - original_elos.min()) / (original_elos.max() - original_elos.min()))[sorting]

    for name in ["Logic", "Clip"]:
        scores = score_dict[name]
    
        # all_elos = np.array([val["elo"] for _, val in scores.items()])[sorting]
        all_elos = np.array([val["elo"] for _, val in scores.items()])
        all_elos = ((all_elos - all_elos.min()) / (all_elos.max() - all_elos.min()))[sorting]

        print(f"Min ELO: {min(all_elos):.2f}\nMax ELO: {max(all_elos):.2f}\nAverage ELO: {sum(all_elos) / len(all_elos):.2f}\nUnique ELOs: {len(set(all_elos))}")
        
        # all_elos_with_noise = [elo + (15 * (np.random.rand()*2 - 1)) for elo in all_elos]


        if name != "Original":
            
            # # For each bin calculate mean and standard deviation
            # bin_means, bin_edges, _ = stats.binned_statistic(range(len(all_elos)), all_elos, statistic='mean', bins=10)
            # bin_std, _, _ = stats.binned_statistic(range(len(all_elos)), all_elos, statistic='std', bins=10)

            # # Compute mean and std of the last half bin and add to means and stds
            # last_mean = np.mean(all_elos[int(np.mean(bin_edges[-2:])):])
            # last_std = np.std(all_elos[int(np.mean(bin_edges[-2:])):])
            # bin_means = np.append(bin_means, last_mean)
            # bin_std = np.append(bin_std, last_std)

            bin_width = 0.1
            bins = 10
            bin_centers = np.arange(0, 1.1, bin_width)
            bin_means = np.zeros(len(bin_centers))
            bin_std = np.zeros(len(bin_centers))
            for i, center in enumerate(bin_centers):
                start = max(0, center - bin_width/2)
                end = min(1, center + bin_width/2)
                mask = (original_elos >= start) & (original_elos < end)
                bin_means[i] = np.mean(all_elos[mask])
                bin_std[i] = np.std(all_elos[mask])

            # Plot fill between the mean and standard deviation
            if name == "Logic":
                ax.fill_between(bin_centers, bin_means - bin_std*2, bin_means + bin_std*2, alpha=0.3, color=COLORS[name], zorder=4)
            else:
                ax.fill_between(bin_centers, bin_means - bin_std*2, bin_means + bin_std*2, alpha=0.3, color=COLORS[name], zorder=3)
        
        # Plot histogram of elos
        # ax.plot(range(len(all_elos)), all_elos, zorder=3, label=name, color=COLORS[name], alpha=0.5)
        if name == "Logic":
            ax.scatter(original_elos, all_elos, zorder=4, label=name, color=COLORS[name], s=2)
        else:
            ax.scatter(original_elos, all_elos, zorder=3, label=name, color=COLORS[name], s=2)
    
    ax.plot([0, 1], [0, 1], color='black', linestyle='--', zorder=6)

    
    ax.legend(loc='upper left')    
    ax.set_xlabel("Normalised ELO (Original)")
    ax.set_ylabel("Normalised ELO (Simulated)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    # plt.title(f"ELO distribution\nMin: {min(all_elos):.2f}, Max: {max(all_elos):.2f}, Average: {sum(all_elos) / len(all_elos):.2f}, Unique: {len(set(all_elos))}")
    plt.tight_layout()
    plt.savefig(f"reports/figures/elo/elo_scatter_{args.name}_new.pdf")
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
    parser.add_argument('--logic', action='store_true', help="Plot logic data")
    parser.add_argument('--clip', action='store_true', help="Plot clip data")
    parser.add_argument('--dataset', default='/full', type=str, help="Folder containing images")
    parser.add_argument('--plot', default='both', choices=['both', 'scores', 'history'], type=str, help="Plot to show (scores or history)")
    args = parser.parse_args()

    score_dict = {}

    scores_file = Path(f'scores/{args.dataset}/elo/{args.name}.json')
    # history_file = Path(f'scores/{os.path.basename(args.folder)}/elo/{args.name}_history.json')

    if not scores_file.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_file}")

    # if not history_file.exists():
    #     raise FileNotFoundError(f"History file not found: {history_file}")
    
    if not os.path.exists('reports/figures/elo'):
        os.makedirs('reports/figures/elo')
    
    with open(scores_file, 'r') as f:
        scores = json.load(f)
    score_dict["Original"] = scores

    if args.logic:
        with open(str(scores_file).replace('.json', '_logic.json'), 'r') as f:
            scores = json.load(f)
        score_dict["Logic"] = scores

    if args.clip:
        with open(str(scores_file).replace('.json', '_clip.json'), 'r') as f:
            scores = json.load(f)
        score_dict["Clip"] = scores

    if args.plot == 'scores' or args.plot == 'both':
        plot_elo_scores(score_dict)
        plot_elo_scores(score_dict, normalized=True)
        plot_elo_scores_scatter(score_dict)
        plot_elo_scores_scatter_new(score_dict)
    
    # with open(history_file, 'r') as f:
    #     history = json.load(f)

    # if args.plot == 'history' or args.plot == 'both':
    #     plot_elo_history(history)