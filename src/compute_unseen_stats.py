import json
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from scipy import stats


def plot_scores():
    with open(f'scores/predictions_100/{args.name}_rated.json') as f:
        scores = json.load(f)
    
    print("User: ", args.name, "\n")


    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Compute the average score for each model and section
    offset = [-0.09, -0.03, 0.03, 0.09]
    i = 0
    for model, model_scores in scores.items():
        print("#"*50)
        print("Model: ", model)
        means = []
        stds = []
        offset_i = offset[i]
        i += 1
        for section, section_scores in model_scores.items():
            print("\tSection: ", section)
            scores = list(section_scores.values())
            print(f"\tAverage score: {np.mean(scores):.1f}, std: {np.std(scores):.2f}")
            means.append(np.mean(scores))
            stds.append(np.std(scores))
        
        ax.errorbar(np.arange(len(means))+offset_i, means, yerr=stds, label=model, fmt='o-')
    
    ax.set_xticks(range(len(means)), ["Worst", "Middle", "Best"])
    ax.set_ylabel("Score")

    ax.legend()
    plt.show()

def get_kendall_tau():
    with open(f'scores/predictions_100/{args.name}_rated.json') as f:
        rated = json.load(f)
    
    with open(f'scores/predictions_100/{args.name}_model_preds.json') as f:
        model_preds = json.load(f)

    print("User: ", args.name, "\n")

    for (model, rated_scores), (_, pred_scores) in zip(rated.items(), model_preds.items()):
        print("#"*50)
        print("Model: ", model)
        r_scores = []
        p_scores = []
        for (section, rated_section_scores), (_, pred_section_scores)  in zip(rated_scores.items(), pred_scores.items()):
            # print("\tSection: ", section)
            r_scores += list(rated_section_scores.values())
            p_scores += list(pred_section_scores.values())

        statistic, p_value = stats.kendalltau(r_scores, p_scores)
        print(f"\tKendall's tau: {statistic:.2f}, p-value: {p_value}")
        print("\n")


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help="Name of the user")

    args = parser.parse_args()

    # Load the scores
    plot_scores()
    get_kendall_tau()

    
