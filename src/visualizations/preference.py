import matplotlib.pyplot as plt
import json
import numpy as np
import os
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser


def show_n_images_in_grid(image_paths: list, n: int=5, pool_size: int=25, position: str="top"):
    assert n**2 <= pool_size, f"n**2 = {n**2} should be less than or equal to the pool size = {pool_size}"

    fig, axes = plt.subplots(n, n, figsize=(10, 10))

    # Choose n**2 random images
    if position == "top":
        index_list = list(range(pool_size))
    elif position == "bottom":
        index_list = list(range(len(image_paths) - pool_size, len(image_paths)))
    else:
        mid_point = int(len(image_paths) / 2)
        index_list = list(range(mid_point - int(pool_size / 2), mid_point + int((pool_size+1) / 2)))
    indeces = np.random.choice(index_list, n**2, replace=False)

    for i, ax in enumerate(axes.flat):
        img = Image.open(image_paths[indeces[i]])
        # resize image
        img = img.resize((224, 224))
        ax.imshow(img)
        ax.axis("off")
    
    fig.suptitle(f"{n**2} random images from the {position} {pool_size} images by ELO rating")
    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("--dataset", type=str, default="full")
    parser.add_argument("--strategy", type=str, default="elo", choices=["elo", "scale"])
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=100)
    args = parser.parse_args()

    if args.strategy == "scale":
        raise NotImplementedError("Scale strategy not implemented yet")
    
    # Load the ratings
    ratings_path = Path(f"scores/{args.dataset}/{args.strategy}/{args.name}.json")
    with open(ratings_path, "r") as f:
        ratings = json.load(f)
    
    # Get the image names
    image_paths = list(f"data/processed/{args.dataset}/{image_name}" for image_name in ratings.keys())

    # Sort the image names by their ratings
    image_paths.sort(key=lambda x: ratings[os.path.basename(x)]["elo"], reverse=True)

    # Show the images
    show_n_images_in_grid(image_paths, args.n, pool_size=args.pool_size, position="bottom")
    show_n_images_in_grid(image_paths, args.n, pool_size=args.pool_size, position="middle")
    show_n_images_in_grid(image_paths, args.n, pool_size=args.pool_size, position="top")