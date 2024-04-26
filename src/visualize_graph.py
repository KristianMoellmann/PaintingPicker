import torch
import torch.nn as nn
import networkx as nx
import netwulf as nw
import json
import matplotlib.pyplot as plt
from custom_data import EmbeddedMatchDataset, EmbeddedMatchDataSplit, MatchDataHistorySplit
from torch.utils.data import DataLoader, random_split
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm


def add_nodes_and_edges(G: nx.DiGraph, history: dict, name: str, color: str, add_edges: bool = True):
    for session, session_history in history.items():
        for time, scoring in session_history.items():
            left_image = scoring["left_image"]
            right_image = scoring["right_image"]
            if not G.has_node(left_image):
                G.add_node(left_image, name=left_image, color=color)
            if not G.has_node(right_image):
                G.add_node(right_image, name=right_image, color=color)
            if add_edges:
                G.add_edge(left_image, right_image, color=color)
    return G


def build_graph(train_history: dict, val_history: dict, test_history: dict, name: str):
    G = nx.DiGraph()

    for history, color, add_edges in zip(
        [train_history, val_history, test_history],
        ['lightblue', 'orange', 'green'],
        [True, False, False]
        ):
        G = add_nodes_and_edges(G, history, name, color, add_edges=add_edges)
    return G


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help="Name of the user")
    parser.add_argument('--dataset', default='full', type=str, help="Folder containing images")
    parser.add_argument('--model', default='ViT-B/32', type=str, help="Model to use")
    parser.add_argument('--seed', default=42, type=int, help="Seed to use for reproducibility")
    parser.add_argument('--split', action='store_true', help="Whether to split the data")
    parser.add_argument('--logic', action='store_true', help='Use the logic model to simulate training data')
    args = parser.parse_args()

    scoring_path = Path(f'scores/{args.dataset}/elo/{args.name}_history.json')

    if not scoring_path.exists():
        raise FileNotFoundError(f"Scoring file {scoring_path} does not exist")
    
    with open(scoring_path, 'r') as f:
        labels = json.load(f)
    
    # Load the data
    history_data = MatchDataHistorySplit(labels)
    _, history_test = history_data.hold_out(ratio=0.1, seed=args.seed)

    history_train, history_val = history_data.hold_out(ratio=0.1, seed=args.seed+1)

    if args.logic:
        # Update history_train using logic to simulate
        pass
    
    
    G = build_graph(history_train, history_val, history_test, args.name)

    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]

    nodes = G.nodes()
    node_colors = [G.nodes[n]['color'] for n in nodes]
    
    new_G = nw.visualize(G)
    # nx.draw(G, edge_color=colors, node_color=node_colors)
    # plt.show()
