import os
import json
import shutil
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm


def create_folder_from_scale_data(scale_data: dict, save_folder: str):
    """
    Create a folder with the images from the scale data

    Args:
        scale_data (dict): The scale data dictionary {"img_id": score}
        save_folder (str): The folder to save the images in
    """
    for img_id, score in tqdm(scale_data.items()):
        img_path = f'data/raw/{img_id}'
        save_path = f'{save_folder}/{img_id}'
        # Copy the image to the save folder on windows
        shutil.copy(img_path, save_path)


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help="Name of the user")
    parser.add_argument('--dataset', default='full', type=str, help="Name of dataset (full, ...)")
    parser.add_argument('--folder', default=None, type=str, help="Folder to save the images in")
    parser.add_argument('--scale', default=9, type=int, help="Number of scales")

    args = parser.parse_args()

    with open(f'scores/{args.dataset}/scale_{args.scale}/{args.name}.json', 'r') as file:
        scale_data = json.load(file)

        if args.folder is None:
            args.folder = f'data/processed/{args.dataset}/{args.name}'
        
        if not os.path.exists(args.folder):
            os.makedirs(args.folder)

        create_folder_from_scale_data(scale_data, args.folder)
