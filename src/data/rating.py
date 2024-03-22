import customtkinter as tk
import random
from glob import glob
from PIL import Image, ImageTk


class Rating(tk.CTk):

    def __init__(self):
        super().__init__()

        self.title('Rating')
        self.geometry('1280x700')

        self.images = glob('data/raw/*.png')
        self.votes = [0, 0]

        # Write the votes as a header
        self.header = tk.CTkLabel(self, text=f'{self.votes[0]} vs {self.votes[1]}', font=('Arial', 20))
        self.header.pack(pady=10)

        # Create two 600x600 canvases side by side but centered
        self.canvas_left = tk.CTkCanvas(self, width=600, height=600)
        self.canvas_left.pack(side=tk.LEFT, padx=(100, 5))
        self.canvas_right = tk.CTkCanvas(self, width=600, height=600)
        self.canvas_right.pack(side=tk.RIGHT, padx=(5, 100))

        self.load_images()

        self.bind('<Left>', self.on_left_key)
        self.bind('<Right>', self.on_right_key)

        # Bind click on image to vote
        self.canvas_left.bind('<Button-1>', self.on_left_key)
        self.canvas_right.bind('<Button-1>', self.on_right_key)

        # Escape key to close the window
        self.bind('<Escape>', lambda e: self.destroy())

    def load_images(self):
        image1_path, image2_path = random.sample(self.images, 2)
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
        self.canvas_left.configure(bg='white')
        self.canvas_right.configure(bg='white')

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
        self.header.configure(text=f'{self.votes[0]} vs {self.votes[1]}')

    def on_left_key(self, event):
        self.canvas_left.configure(bg='green')
        self.votes[0] += 1
        self.update_header()
        self.load_images()
    
    def on_right_key(self, event):
        self.canvas_right.configure(bg='green')
        self.votes[1] += 1
        self.update_header()
        self.load_images()
    
    def run(self):
        self.mainloop()

if __name__ == '__main__':
    app = Rating()
    app.run()