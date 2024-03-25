import customtkinter as tk
import random
from glob import glob
from PIL import Image, ImageTk
from argparse import ArgumentParser
from pathlib import Path
import os
import json


class Scale(tk.CTk):

    def __init__(self, name: str, folder: str, scale: int = 9):
        super().__init__()
        self.name = name
        self.folder = folder
        self.scores_folder = Path(f'scores/{os.path.basename(folder)}/scale_{scale}')

        self.title('Rating')
        self.geometry('1280x720')

        self.images = glob(f'{folder}/*.jpg')
        self.images.sort()
        
        self.number_of_images = len(self.images)

        self.scores = self.load_image_scores()

        self.image_index = len(self.scores)

        # Write the votes as a header
        self.header = tk.CTkLabel(self, text=f'Painting {self.image_index + 1} of {self.number_of_images}', font=('Arial', 20))
        self.header.pack(pady=10)

        # Create a 1000x600 canvas
        self.canvas = tk.CTkCanvas(self, width=1000, height=600)
        self.canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.load_image()
        
        if scale == 2:
            # Bind 1-scale keys to vote
            self.bind('1', lambda e: self.on_key(0))
            self.bind('2', lambda e: self.on_key(1))

            # Bind left and right
            self.bind('<Left>', lambda e: self.on_key(0))
            self.bind('<Right>', lambda e: self.on_key(1))
            
            # Add button for each scale at the bottom
            button = tk.CTkButton(self, text="Don't like", command=lambda: self.on_key(0))
            button.place(relx=0.4, rely=0.95, anchor=tk.CENTER)
            button = tk.CTkButton(self, text="Like", command=lambda: self.on_key(1))
            button.place(relx=0.6, rely=0.95, anchor=tk.CENTER)
        else:
            # Bind 1-scale keys to vote
            for i in range(1, scale + 1):
                self.bind(str(i), self.on_key)
                
                # Add button for each scale at the bottom
                button = tk.CTkButton(self, text=str(i), command=lambda i=i: self.on_key(i))
                button.place(relx=i/(scale+1), rely=0.95, anchor=tk.CENTER, relwidth=1/(scale+2))

        # Escape key to close the window
        self.bind('<Escape>', lambda e: self.destroy())
    
    def load_image_scores(self):
        scores = {}
        if not os.path.exists(self.scores_folder):
            os.makedirs(self.scores_folder, exist_ok=True)
        scores_file = f"{self.scores_folder}/{self.name}.json"
        if not os.path.exists(scores_file):
            with open(scores_file, 'w') as f:
                json.dump(scores, f)
        else:
            with open(scores_file, 'r') as f:
                scores = json.load(f)
        return scores

    def save_image_scores(self):
        scores_file = f"{self.scores_folder}/{self.name}.json"
        with open(scores_file, 'w') as f:
            json.dump(self.scores, f)
    
    def load_image(self):
        if self.image_index >= self.number_of_images:
            self.destroy()
            return
        image_path = self.images[self.image_index]
        image = Image.open(image_path)
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
        image_name = os.path.basename(self.images[self.image_index])
        if isinstance(event, int):
            self.scores[image_name] = event
        else:
            self.scores[image_name] = int(event.char)
        self.image_index += 1
        self.save_image_scores()
        self.load_image()

    def run(self):
        self.mainloop()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help="Name of the user")
    parser.add_argument('--folder', default='data/fewer_imgs', type=str, help="Folder containing images")
    parser.add_argument('--scale', type=int, default=9, help="Number of scales")
    args = parser.parse_args()
    app = Scale(args.name, args.folder, args.scale)
    app.run()