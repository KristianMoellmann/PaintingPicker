import customtkinter as tk
import random
from glob import glob
from PIL import Image, ImageTk
from argparse import ArgumentParser
from pathlib import Path
import os
import json

class Rating(tk.CTk):

    def __init__(self, name: str, folder: str):
        super().__init__()
        self.name = name
        self.folder = folder
        self.scores_folder = Path(f'reports/scores/{os.path.basename(folder)}')

        self.title('Rating')
        self.geometry('1280x720')

        self.images = glob(f'{folder}/*.jpg')
        self.left_image_name = None
        self.right_image_name = None

        self.match_index = 0
        self.image_pairings = self.restart_round()
        self.number_of_pairings = len(self.image_pairings)

        # Write the votes as a header
        self.header = tk.CTkLabel(self, text=f'Match {1} of {self.number_of_pairings}', font=('Arial', 20))
        self.header.pack(pady=10)

        # Create two 600x600 canvases side by side but centered
        self.canvas_left = tk.CTkCanvas(self, width=600, height=600)
        self.canvas_left.configure(bg='black')
        self.canvas_left.place(relx=0.25, rely=0.5, anchor=tk.CENTER)
        self.canvas_right = tk.CTkCanvas(self, width=600, height=600)
        self.canvas_right.configure(bg='black')
        self.canvas_right.place(relx=0.75, rely=0.5, anchor=tk.CENTER)

        self.scores = self.load_image_scores()
        self.load_images()

        self.bind('<Left>', self.on_left_key)
        self.bind('<Right>', self.on_right_key)

        # Bind click on image to vote
        self.canvas_left.bind('<Button-1>', self.on_left_key)
        self.canvas_right.bind('<Button-1>', self.on_right_key)

        # Escape key to close the window
        self.bind('<Escape>', lambda e: self.destroy())
    
    def create_image_pairings(self):
        # Pair each image with another image randomly. If odd number, one image will have two pairings
        image_pairings = []
        images = self.images.copy()
        while len(images) > 1:
            image1 = images.pop(random.randint(0, len(images)-1))
            image2 = images.pop(random.randint(0, len(images)-1))
            image_pairings.append((image1, image2))
        if len(images) == 1:
            random_image = random.choice(self.images)
            while random_image == images[0]:
                random_image = random.choice(self.images)
            image_pairings.append((images[0], random_image))
        return image_pairings
    
    def load_image_scores(self):
        # Load image scores from json file...
        # If score folder does not exist, create it
        if not self.scores_folder.exists():
            self.scores_folder.mkdir()
        # If scores file does not exist, create it
        scores_file = Path(f'{self.scores_folder}/{self.name}.json')
        if not scores_file.exists():
            with open(scores_file, 'w') as f:
                # TODO: Change to include the necessary metrics
                scores = {}
                for image in self.images:
                    scores[os.path.basename(image)] = 0
                f.write(json.dumps(scores))
        else:
            with open(scores_file, 'r') as f:
                scores = json.load(f)
        sum = 0
        for score in scores.values():
            sum += score
        print(f'Loaded scores for {self.name} with a total of {sum} votes')
        return scores
    
    def save_image_scores(self):
        scores_file = Path(f'{self.scores_folder}/{self.name}.json')
        with open(scores_file, 'w') as f:
            f.write(json.dumps(self.scores))
    
    def restart_round(self):
        self.match_index = 0
        return self.create_image_pairings()

    def load_images(self):
        if self.match_index >= self.number_of_pairings:
            self.image_pairings = self.restart_round()
        image1_path, image2_path = self.image_pairings[self.match_index]
        self.match_index += 1

        self.left_image_name = os.path.basename(image1_path)
        self.right_image_name = os.path.basename(image2_path)

        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)
        # Resize images to fit the canvas such that the largest dimension is 600 but keeps ratio
        image1 = self.resize_image(image1, 600)
        image2 = self.resize_image(image2, 600)
        # Resize canvas to fit the image
        self.canvas_left.configure(width=image1.width, height=image1.height)
        self.canvas_right.configure(width=image2.width, height=image2.height)
        # Display images
        self.image1 = ImageTk.PhotoImage(image1)
        self.image2 = ImageTk.PhotoImage(image2)
        self.canvas_left.create_image(0, 0, image=self.image1, anchor=tk.NW)
        self.canvas_right.create_image(0, 0, image=self.image2, anchor=tk.NW)
        # Reset canvas background
        self.update_header()

    def resize_image(self, image, max_size):
        width, height = image.width, image.height
        if width > height:
            new_width = max_size
            new_height = int(max_size * height / width)
        else:
            new_height = max_size
            new_width = int(max_size * width / height)
        return image.resize((new_width, new_height))

    def update_header(self):
        self.header.configure(text=f'Match {self.match_index} of {self.number_of_pairings}')

    def on_left_key(self, event):
        self.canvas_left.configure(bg='green')
        self.scores[self.left_image_name] += 1 # TODO: Change to a function which calculates the accurate ELO between two images
        self.save_image_scores()
        self.load_images()
    
    def on_right_key(self, event):
        self.canvas_right.configure(bg='green')
        self.scores[self.right_image_name] += 1
        self.save_image_scores()
        self.load_images()
    
    def run(self):
        self.mainloop()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("name", default=None, type=str, help="Name of the user")
    parser.add_argument("--folder", default='data/fewer_imgs', type=str, help="Folder containing images")
    args = parser.parse_args()

    app = Rating(args.name, args.folder)
    app.run()