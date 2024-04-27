import json
import numpy as np
from argparse import ArgumentParser





if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help="Name of the user")

    args = parser.parse_args()

    # Load the scores

    with open(f'scores/predictions/{args.name}_rated.json') as f:
        scores = json.load(f)
    
    print("User: ", args.name, "\n")
    
    # Compute the average score for each model and section
    for model, model_scores in scores.items():
        print("#"*50)
        print("Model: ", model)
        for section, section_scores in model_scores.items():
            print("\tSection: ", section)
            scores = list(section_scores.values())
            print(f"\tAverage score: {np.mean(scores):.1f}, std: {np.std(scores):.2f}")
    
