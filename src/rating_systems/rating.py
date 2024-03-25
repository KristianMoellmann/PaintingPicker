import customtkinter as tk
import random
from glob import glob
from PIL import Image, ImageTk
from argparse import ArgumentParser
from pathlib import Path
import os
import json
from datetime import datetime
from tqdm import tqdm

class Rating(tk.CTk):

    def __init__(self, name: str, folder: str, strategy: str = 'random'):
        super().__init__()
        self.name = name
        self.strategy = strategy
        self.folder = folder
        self.scores_folder = Path(f'scores/{os.path.basename(folder)}/elo')
        self.session = None

        self.title('Rating')
        self.geometry('1280x720')

        self.images = glob(f'{folder}/*.jpg')
        self.images.sort()
        self.left_image_name = None
        self.right_image_name = None

        # Create two 600x600 canvases side by side but centered
        self.canvas_left = tk.CTkCanvas(self, width=600, height=600)
        self.canvas_left.place(relx=0.25, rely=0.5, anchor=tk.CENTER)
        self.canvas_right = tk.CTkCanvas(self, width=600, height=600)
        self.canvas_right.place(relx=0.75, rely=0.5, anchor=tk.CENTER)

        self.bind('<Up>', self.on_up_key)
        self.bind('<Left>', self.on_left_key)
        self.bind('<Right>', self.on_right_key)

        # Bind click on image to vote
        self.canvas_left.bind('<Button-1>', self.on_left_key)
        self.canvas_right.bind('<Button-1>', self.on_right_key)

        # Escape key to close the window
        self.bind('<Escape>', lambda e: self.destroy())

        # Load the scores
        self.scores, self.history = self.load_image_elo_scores()
        self.match_index = 0
        self.image_pairings = self.restart_round(self.strategy)
        self.number_of_pairings = len(self.image_pairings)

        # Write the votes as a header
        self.header = tk.CTkLabel(self, text=f'Match {1} of {self.number_of_pairings}', font=('Arial', 20))
        self.header.pack(pady=10)

        self.load_images()

    
    def create_random_image_pairings(self):
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
    
    
    def create_smart_image_pairings(self):
        filename_to_fullpath = {os.path.basename(path): path for path in self.images}
        images_sorted_by_elo_filenames = sorted(self.scores, key=lambda x: self.scores[x]['elo'])
        images_sorted_by_elo = [filename_to_fullpath[filename] for filename in images_sorted_by_elo_filenames]

        image_pairings = []
        pbar = tqdm(total=len(images_sorted_by_elo))
        while len(images_sorted_by_elo) > 1:
            image1 = min(images_sorted_by_elo, key=lambda x: self.scores[os.path.basename(x)]['matches'])
            pbar.update(1)
            images_sorted_by_elo.remove(image1)
            index_of_image1 = images_sorted_by_elo_filenames.index(os.path.basename(image1))
            
            start_index = max(0, index_of_image1 - 15)
            end_index = min(len(images_sorted_by_elo_filenames), index_of_image1 + 16)  # +16 to include the 15th index on the right
            possible_match_filenames = images_sorted_by_elo_filenames[start_index:index_of_image1] + images_sorted_by_elo_filenames[index_of_image1 + 1:end_index]
            possible_matches = [filename_to_fullpath[filename] for filename in possible_match_filenames if filename_to_fullpath[filename] in images_sorted_by_elo]

            if possible_matches:
                # Select a random image from the possible matches
                image2 = random.choice(possible_matches)
                images_sorted_by_elo.remove(image2)
            else:
                # If there are no possible matches left break
                break

            image_pairings.append((image1, image2))
            pbar.update(1)
        return image_pairings

    
    def load_image_elo_scores(self):
        # Load image scores from json file...
        self.scores_folder.mkdir(parents=True, exist_ok=True)
        # Load elo scores from json file
        elo_scores_file = Path(f'{self.scores_folder}/{self.name}.json')
        if not elo_scores_file.exists():
            scores = {
                os.path.basename(image): {"elo": 1400, "matches": 0, "wins": 0, "losses": 0, "draws": 0}
                for image in self.images
            }
            with open(elo_scores_file, 'w') as f:
                f.write(json.dumps(scores, indent=4))
        else:
            with open(elo_scores_file, 'r') as f:
                scores = json.load(f)
        
        # Load match up history from json file
        history_file = Path(f'{self.scores_folder}/{self.name}_history.json')
        if not history_file.exists():
            history = {"0": {}}
            self.session = "0"
            with open(history_file, 'w') as f:
                f.write(json.dumps(history, indent=4))
        else:
            with open(history_file, 'r') as f:
                history = json.load(f)
            self.session = str(max(map(int, history.keys())) + 1)
            history[self.session] = {}
        
        return scores, history
    
    
    def save_image_scores(self):
        scores_file = Path(f'{self.scores_folder}/{self.name}.json')
        with open(scores_file, 'w') as f:
            f.write(json.dumps(self.scores, indent=4))
        
        history_file = Path(f'{self.scores_folder}/{self.name}_history.json')
        with open(history_file, 'w') as f:
            f.write(json.dumps(self.history, indent=4))

    
    def restart_round(self, strategy):
        self.match_index = 0
        if strategy == "smart":
            return self.create_smart_image_pairings()
        else:
            return self.create_random_image_pairings()


    def load_images(self):
        if self.match_index >= self.number_of_pairings:
            self.image_pairings = self.restart_round(self.strategy)
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


    def calculate_elo(self, ra, rb, sa, sb, K=32):
        """
        Calculate the new Elo ratings for two players.

        Parameters:
        ra (float): The current rating of player A.
        rb (float): The current rating of player B.
        sa (float): The score of player A (1 for win, 0.5 for draw, 0 for loss).
        sb (float): The score of player B (1 for win, 0.5 for draw, 0 for loss).
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
    

    def on_left_key(self, event):
        self.canvas_left.configure(bg='green')
        
        # Extract current Elo scores
        left_score = self.scores[self.left_image_name]["elo"]
        right_score = self.scores[self.right_image_name]["elo"]
        
        # Calculate new Elo scores with left image as winner
        left_score_new, right_score_new = self.calculate_elo(left_score, right_score, 1, 0)
        
        # Update scores and record the match
        self.scores[self.left_image_name]["elo"] = left_score_new
        self.scores[self.left_image_name]["matches"] += 1
        self.scores[self.left_image_name]["wins"] += 1
        
        self.scores[self.right_image_name]["elo"] = right_score_new
        self.scores[self.right_image_name]["matches"] += 1
        self.scores[self.right_image_name]["losses"] += 1

        # store history at current datetime
        self.history[self.session][datetime.now().isoformat()] = {
            "left_image": self.left_image_name,
            "right_image": self.right_image_name,
            "winner": 0
        }
        
        self.save_image_scores()
        self.load_images()

    
    def on_right_key(self, event):
        self.canvas_left.configure(bg='green')
        
        # Extract current Elo scores
        left_score = self.scores[self.left_image_name]["elo"]
        right_score = self.scores[self.right_image_name]["elo"]
        
        # Calculate new Elo scores with left image as winner
        left_score_new, right_score_new = self.calculate_elo(left_score, right_score, 0, 1)
        
        # Update scores and record the match
        self.scores[self.left_image_name]["elo"] = left_score_new
        self.scores[self.left_image_name]["matches"] += 1
        self.scores[self.left_image_name]["losses"] += 1
        
        self.scores[self.right_image_name]["elo"] = right_score_new
        self.scores[self.right_image_name]["matches"] += 1
        self.scores[self.right_image_name]["wins"] += 1

        # store history at current datetime
        self.history[self.session][datetime.now().isoformat()] = {
            "left_image": self.left_image_name,
            "right_image": self.right_image_name,
            "winner": 1
        }
        
        self.save_image_scores()
        self.load_images()
        
        
    def on_up_key(self, event):
        self.canvas_left.configure(bg='green')
        
        # Extract current Elo scores
        left_score = self.scores[self.left_image_name]["elo"]
        right_score = self.scores[self.right_image_name]["elo"]
        
        # Calculate new Elo scores with left image as winner
        left_score_new, right_score_new = self.calculate_elo(left_score, right_score, 0.5, 0.5)
        
        # Update scores and record the match
        self.scores[self.left_image_name]["elo"] = left_score_new
        self.scores[self.left_image_name]["matches"] += 1
        self.scores[self.left_image_name]["draws"] += 1
        
        self.scores[self.right_image_name]["elo"] = right_score_new
        self.scores[self.right_image_name]["matches"] += 1
        self.scores[self.right_image_name]["draws"] += 1

        # store history at current datetime
        self.history[self.session][datetime.now().isoformat()] = {
            "left_image": self.left_image_name,
            "right_image": self.right_image_name,
            "winner": 2
        }
        
        self.save_image_scores()
        self.load_images()
        
    
    def run(self):
        self.mainloop()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("name", default=None, type=str, help="Name of the user")
    parser.add_argument("--folder", default='data/fewer_imgs', type=str, help="Folder containing images")
    parser.add_argument("--strategy", default='random', choices=["random", "smart"], type=str, help="Strategy for selecting images")
    args = parser.parse_args()
    
    app = Rating(args.name, args.folder, args.strategy)
    app.run()