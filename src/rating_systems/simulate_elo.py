# Use the transitative properties of the Elo rating system to simulate new matches.
# However we would might see that A < B < C < D < E < C
# This is a cyclic dependency and we need to handle it. 

import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
import shutil
from datetime import datetime


def load_match_data(filename):
    """ Load match history from a JSON file. """
    with open(filename, 'r') as file:
        return json.load(file)


def initialize_dictionarioary(match_data, image_names):
    # Initialize the dictionary for tracking wins and losses
    init_wins_and_loses = {}
    for image_name in image_names.keys():
        init_wins_and_loses[image_name] = {"W": set(), "L": set()}

    # Go trough each session which actually happend
    for session in match_data.keys():
        for match in match_data[session]:
            
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


def plot_unique_wins(pure_wins, path_name):
    # How many diffrent values are there
    all_values = []
    for values in pure_wins.values():
        all_values.append(len(values))
        
    all_values_set = set(all_values)

    # Create a histogram of the values
    plt.hist(all_values, range=[0, 700], bins=700, color='blue', edgecolor='black') 

    # Add titles and labels
    plt.title(f'Histogram of the {len(all_values_set)} unique values for {path_name}')
    plt.xlabel('Value Range')
    plt.ylabel('Count')

    # Show the plot
    plt.show()


def create_upcoming_match_list(pure_wins):
    upcoming_matches = []
    for image in pure_wins.keys():
        for loser in pure_wins[image]:
            upcoming_matches.append((image, loser))
            
    # now we randomly shuffle the matches (TODO: implement a better way to make order of matches)
    random.shuffle(upcoming_matches)
    return upcoming_matches


def calculate_elo(ra, rb, sa, sb, K=32):
    """
    Calculate the new Elo ratings for two players.

    Parameters:
    ra (float): The current rating of player A.
    rb (float): The current rating of player B.
    sb (float): The score of player B (1 for win, 0.5 for draw, 0 for loss).
    sa (float): The score of player A (1 for win, 0.5 for draw, 0 for loss).
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
    

def simulate_matches(upcoming_matches, scores):
    
    # Idea! After every 100 match the order of the matches should be sorted 
    # according to how many matches they have had 
    for match in upcoming_matches:
            winner = match[0]
            loser = match[1]
            winner_score = scores[winner]["elo"]
            loser_score = scores[match[1]]["elo"]

            winner_score_new, loser_score_new = calculate_elo(winner_score, loser_score, 1, 0)
            
            # Update scores and record the match
            scores[winner]["elo"] = winner_score_new
            scores[winner]["matches"] += 1
            scores[winner]["wins"] += 1
            
            scores[loser]["elo"] = loser_score_new
            scores[loser]["matches"] += 1
            scores[loser]["losses"] += 1
            
    return scores
    

def save_image_scores(new_scores, path_name, original_path):
    scores_file = Path(f'scores/full/elo/{path_name}_logic.json')
    if not scores_file.exists():
        # Create the directory if it doesn't exist
        scores_file.parent.mkdir(parents=True, exist_ok=True)
        # Copy the original file to the new location
        shutil.copyfile(original_path, scores_file)
    with open(scores_file, 'w') as f:
        f.write(json.dumps(new_scores, indent=4))
                

# TODO this might work
def save_match_history(match_history, upcoming_matches, path_name, original_history_path):
    sessions = list(match_history.keys())
    new_session = int(max(sessions)) +  1
    match_history[new_session] = {}

    for match in upcoming_matches:
        winner = match[0]
        loser = match[1]
        
        match_history[new_session][datetime.now().isoformat()] = {
            "left_image": winner,
            "right_image": loser,
            "winner": 0
            }
    
    history_file = Path(f'scores/full/elo/{path_name}_logic_history.json')
    if not history_file.exists():
        # Create the directory if it doesn't exist
        history_file.parent.mkdir(parents=True, exist_ok=True)
        # Copy the original file to the new location
        shutil.copyfile(original_history_path, history_file)
    with open(history_file, 'w') as f:
        f.write(json.dumps(match_history, indent=4))


def main():
    for path_name in ["kasper", "kristoffer", "kristoffer_r", "Kristian", "darkness", "darkness_r", "darkness_2500", "darkness_r_2500"]:
        # Load data
        original_path = f'scores/full/elo/{path_name}.json'
        image_names_with_elo = load_match_data(original_path)
        
        # TODO implement test set
        
        original_history_path = f'scores/full/elo/{path_name}_history.json'
        match_history = load_match_data(f'scores/full/elo/{path_name}_history.json')
        
        # find pure wins
        init_wins_and_loses = initialize_dictionarioary(match_history, image_names_with_elo)
        pure_wins = find_transitative_wins_and_loses(init_wins_and_loses)
        # plot_unique_wins(pure_wins, path_name)
        
        # simulate the elo rating system
        upcoming_matches = create_upcoming_match_list(pure_wins)
        new_scores = simulate_matches(upcoming_matches, image_names_with_elo)
        
        # save the new scores
        save_image_scores(new_scores, path_name, original_path)
        save_match_history(match_history, upcoming_matches, path_name, original_history_path)
        

if __name__ == "__main__":
    main()
