import tkinter as tk
import numpy as np


class MousePlotter:
    def __init__(self, root, width=600, height=400):
        self.root = root
        self.width = width
        self.height = height

        # Canvas for drawing
        self.canvas = tk.Canvas(root, width=width, height=height, bg='white')
        self.canvas.pack(pady=10)

        # Bind mouse drag event
        self.canvas.bind("<B1-Motion>", self.paint)

        # Dictionary to store the lowest/latest y-value for each unique x coordinate
        self.raw_points = {}

        # Done button
        self.btn = tk.Button(root, text="Finish & Sample Array", command=self.process_data)
        self.btn.pack(pady=5)

    def paint(self, event):
        x, y = event.x, event.y
        if 0 <= x < self.width and 0 <= y < self.height:
            # Draw a small dot where the mouse is
            self.canvas.create_oval(x - 1, y - 1, x + 1, y + 1, fill="black", width=2)
            # Invert Y so that the bottom of the screen is 0, not the top
            self.raw_points[x] = self.height - y

    def process_data(self):
        if not self.raw_points:
            print("No data drawn!")
            return

        # Get the range of X drawn by the user
        all_x = sorted(self.raw_points.keys())
        min_x, max_x = all_x[0], all_x[-1]

        # Create a fixed interval grid for X (e.g., every 5 pixels)
        step = 5
        sampled_x = np.arange(min_x, max_x + step, step)
        sampled_y = []

        for x in sampled_x:
            # If the exact x wasn't sampled, find the nearest neighbor's y-value
            closest_x = min(self.raw_points.keys(), key=lambda k: abs(k - x))
            sampled_y.append(self.raw_points[closest_x])

        sampled_y_array = np.array(sampled_y)

        print("\n--- Processing Complete ---")
        print(f"Sampled {len(sampled_y_array)} points at fixed intervals of {step} pixels.")
        print("Y-Values Array:\n", sampled_y_array)

        self.root.destroy()


# Run the interactive canvas
root = tk.Tk()
root.title("Draw your function")
app = MousePlotter(root)
root.mainloop()