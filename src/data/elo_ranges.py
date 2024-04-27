import json
import os

def get_elo_range(filename):
    with open(os.path.join(path_to_dir, filename), 'r') as f:
        data = json.load(f)
        elos = [val["elo"] for val in data.values()]
    return elos

if __name__ == "__main__":
    path_to_dir = 'scores/full/elo'
    names = ["kasper", "Kristian", "Kristoffer"]
    add_ons = ["", "_logic", "_clip"]
    for add_on in add_ons:
        print(f"Add-on: {add_on.replace('_', '')}")
        add_on_min_elos = []
        add_on_max_elos = []
        for name in names:
            filename = f"{name}{add_on}.json"
            elos = get_elo_range(filename)
            print(f"    {name: >10}:    Min: {min(elos):.2f}, Max: {max(elos):.2f}")
            add_on_min_elos.append(min(elos))
            add_on_max_elos.append(max(elos))
        print("    -----------------------------------------")
        print(f"                   Min: {min(add_on_min_elos):.2f}, Max: {max(add_on_max_elos):.2f}\n")