# Use the transitative properties of the Elo rating system to simulate new matches.
# However we would might see that A < B < C < D < E < C
# This is a cyclic dependency and we need to handle it. 

import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


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
        while win_search_list:
            beaten_image = win_search_list.pop()
            if beaten_image not in image_wins_set:
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



def main():
    for path_name in ["kasper", "kristoffer", "Kristian", "darkness", "darkness_2500"]:
        image_names = load_match_data(f'scores/full/elo/{path_name}.json')
        match_data = load_match_data(f'scores/full/elo/{path_name}_history.json')
        init_wins_and_loses = initialize_dictionarioary(match_data, image_names)
        pure_wins = find_transitative_wins_and_loses(init_wins_and_loses)
        plot_unique_wins(pure_wins, path_name)


if __name__ == "__main__":
    main()
