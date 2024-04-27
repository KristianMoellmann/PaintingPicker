# Python src/rating_systems/simulate_elo.py Kristian --data unseen
# Python src/rating_systems/rating.py Kristian --folder data/processed/unseen
# Python src/train_model.py Kristian --scoring elo --score_type logic --dont_plot
# Python src/test_model.py Kristian --scoring elo --score_type logic --model model1 --plot True

import argparse
import subprocess

def execute_command(command):
    """Execute a single command in the shell and capture its output."""
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print(f"Output for {command}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {command}: {e}\n{e.stderr}")


def main(name, scoring, score_type, model, plot):
    """Run specified commands for a given name."""
    commands = [
        # f"Python src/data/make_dataset.py unseen",
        # f"Python src/data/embed_dataset.py unseen",
        f"Python src/rating_systems/rating.py {name} --folder data/processed/unseen",
        f"Python src/rating_systems/simulate_elo.py {name} --data unseen",
        # f"Python src/train_model.py {name} --scoring {scoring} --score_type {score_type} --dont_plot",
        # f"Python src/test_model.py {name} --scoring {scoring} --score_type {score_type} --model {model} --plot {plot}"
    ]

    for command in commands:
        print("--------------------")
        print(f"Executing command: {command}")
        if command[26:35] == "rating.py":
            # The rating.py is just used to create the .json files for the rating system
            print("Just close the pop-up window imiddeatly after it opens.")
        
        execute_command(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute specific operations for a given name.")
    parser.add_argument('name', type=str, help="Name to use in commands")
    parser.add_argument("--scoring", default="elo", type=str, choices=["elo", "scale_9"], help="Scoring method to use")
    # parser.add_argument("--data", default="unseen", type=str, choices=["full", "unseen"], help="Data to use")
    parser.add_argument("--score_type", default="logic", type=str, choices=["original", "logic", "clip"], help="Score type to use")
    parser.add_argument("--model", default="clip", type=str, choices=["clip", "logic", "original", "scale_9"], help="Model to use")
    parser.add_argument('--plot', default=False, type=bool, help="Plot the predictions")

    args = parser.parse_args()
    
    main(args.name, args.scoring, args.score_type, args.model, args.plot)
    
    # Example usage:
    # Python src/get_test_results.py kasper --scoring elo --score_type logic --model non --plot False
    