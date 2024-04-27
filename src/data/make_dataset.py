import argparse
import shutil
from glob import glob
import json


def extract_data(dataset):
    all_images = glob(f'data/raw/*.jpg')
    if dataset == 'micro':
        print('Extracting micro dataset')
        images = all_images[:3]
    elif dataset == 'tiny':
        print('Extracting tiny dataset')
        images = all_images[:10]
    elif dataset == 'full':
        print('Extracting full dataset')
        images = all_images[:1000]
    elif dataset == "unseen":
        print('Extracting unseen dataset')
        
        # Get the images that have been seen before
        seen_before_kasper = 'scores/full/scale_9/kasper.json'
        with open(seen_before_kasper, 'r') as f:
            kasper_before = json.load(f)
        
        seen_before_kristoffer = 'scores/full/scale_9/kristoffer.json'
        with open(seen_before_kristoffer, 'r') as f:
            kristoffer_before = json.load(f)
        
        pictures_seen_before = set(list(kasper_before.keys()) + list(kristoffer_before.keys()))
        images = []
        
        count = 0
        for image in all_images:
            if image[9:] not in pictures_seen_before:
                images.append(image)
                count += 1
                if count == 1000:
                    break
                
        
    shutil.rmtree(f'data/processed/{dataset}', ignore_errors=True)
    shutil.os.makedirs(f'data/processed/{dataset}')

    for image in images:
        shutil.copy(image, f'data/processed/{dataset}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['micro', 'tiny', 'full', "unseen"], help='Choose between micro, tiny, full and unseen dataset (3 vs. 10 vs. 1000 images)')
    args = parser.parse_args()
    extract_data(args.dataset)
