import torch
import clip
import os
import glob
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from PIL import Image


def embed_images(folder: str, model: str, device: str) -> None:
    # Load the model
    device = torch.device(device)
    model, preprocess = clip.load(model, device=device)
    model.float()

    # Get the list of image files
    image_files = glob.glob(os.path.join(folder, '*.jpg'))

    # Create the output folder
    output_folder = os.path.join(f"data/embedded/{os.path.basename(folder)}")
    os.makedirs(output_folder, exist_ok=True)

    # Embed the images
    for image_file in tqdm(image_files):
        image = preprocess(Image.open(image_file)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        image_features = image_features.cpu()
        output_file = os.path.join(output_folder, os.path.basename(image_file).replace('.jpg', '.pt'))
        with open(output_file, 'wb') as f:
            torch.save(image_features, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str, choices=["micro", "tiny", "full"], help="Folder containing images")
    parser.add_argument('--model', default='ViT-B/32', type=str, help="Model to use")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_images(f"data/processed/{args.dataset}", args.model, device)
