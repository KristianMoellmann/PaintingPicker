import customtkinter as tk
import random
from glob import glob
from PIL import Image, ImageTk
from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict
import os
import json


class Scale(tk.CTk):

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.folder = "data/raw/"
        if os.path.exists(f'scores/predictions_100/{name}_rated.json'):
            self.scores_path = Path(f'scores/predictions_100/{name}_rated.json')
        else:
            self.scores_path = Path(f'scores/predictions_100/{name}.json') # TODO changed to predictions_100
        self.old_scores_path = Path(f'scores/predictions/{name}_rated.json')

        self.title('Rating')
        self.geometry('1280x720')

        self.images, self.scores, self.image_to_model = self.load_images_and_scores(self.scores_path)
        
        self.number_of_images = len(self.images)

        self.image_index = 0

        # Write the votes as a header
        self.header = tk.CTkLabel(self, text=f'Painting {self.image_index + 1} of {self.number_of_images}', font=('Arial', 20))
        self.header.pack(pady=10)

        # Create a 1000x600 canvas
        self.canvas = tk.CTkCanvas(self, width=1000, height=600)
        self.canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.load_image()
        

        # Bind 1-scale keys to vote
        for i in range(1, 10):
            self.bind(str(i), self.on_key)
            
            # Add button for each scale at the bottom
            button = tk.CTkButton(self, text=str(i), command=lambda i=i: self.on_key(i))
            button.place(relx=i/10, rely=0.95, anchor=tk.CENTER, relwidth=1/10)

        # Escape key to close the window
        self.bind('<Escape>', lambda e: self.destroy())
    
    def load_images_and_scores(self, scores_path):
        with open(scores_path, 'r') as f:
            scores = json.load(f)

        with open(self.old_scores_path, 'r') as f:
            old_scores = json.load(f)

        old_ratings = {}
        for model, model_scores in old_scores.items():
            for section, section_scores in model_scores.items():
                for image, score in section_scores.items():
                    old_ratings[image] = score

        # Get set of all images
        images = set()
        image_to_model = defaultdict(lambda: [])
        scoring = {}
        for model, model_scores in scores.items():
            scoring[model] = {}
            for section, section_scores in model_scores.items():
                scoring[model][section] = {}
                for image, score in section_scores.items():
                    if score != 0 and score != 0.1:
                        scoring[model][section][image] = score
                        continue
                    
                    if image in old_ratings:
                        scoring[model][section][image] = old_ratings[image]
                    else:
                        scoring[model][section][image] = 0
                        images.add(image)
                        image_to_model[image].append((model, section))

        images = list(images)
        # Randomise the order of the images
        random.shuffle(images)
        return images, scoring, image_to_model

    def update_scores(self, image_name, score):
        for model, section in self.image_to_model[image_name]:
            self.scores[model][section][image_name] = score

    def save_image_scores(self):
        scores_file = Path(f'scores/predictions_100/{self.name}_rated.json')
        with open(scores_file, 'w') as f:
            json.dump(self.scores, f, indent=4)
    
    def load_image(self):
        if self.image_index >= self.number_of_images:
            self.save_image_scores()
            self.destroy()
            return
        
        image_path = self.images[self.image_index]
        image = Image.open(self.folder + image_path)
        image = self.resize_image(image, 1000, 600)
        self.canvas.configure(width=image.width, height=image.height)
        self.image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=self.image, anchor=tk.NW)
        self.update_header()

    def resize_image(self, image, max_width, max_height):
        # Scale but keep aspect ratio
        width, height = image.size
        aspect_ratio = width / height
        if width > max_width:
            width = max_width
            height = int(width / aspect_ratio)
        if height > max_height:
            height = max_height
            width = int(height * aspect_ratio)
        return image.resize((width, height))

    def update_header(self):
        self.header.configure(text=f'Match {self.image_index + 1} of {self.number_of_images}')

    def on_key(self, event):
        image_name = self.images[self.image_index]
        score = event if isinstance(event, int) else int(event.char)
        self.update_scores(image_name, score)
        self.image_index += 1
        self.save_image_scores()
        self.load_image()

    def run(self):
        self.mainloop()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help="Name of the user")
    args = parser.parse_args()
    app = Scale(args.name)
    app.run()