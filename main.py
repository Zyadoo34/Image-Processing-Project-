import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2

import operations

# cooment
class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Application")

        self.original_img = None
        self.processed_img = None
        self.active_filter = None  # <-- key idea

        # ---------- Buttons ----------
        tk.Button(root, text="Upload Image", command=self.upload_image).pack()
        tk.Button(root, text="Grayscale", command=self.apply_grayscale).pack()
        tk.Button(root, text="Histogram Equalization", command=self.apply_hist).pack()
        tk.Button(root, text="Gaussian Blur", command=self.select_blur).pack()
        tk.Button(root, text="Sharpen", command=self.select_sharpen).pack()
        tk.Button(root, text="Threshold", command=self.select_threshold).pack()
        tk.Button(root, text="Erosion", command=self.select_erosion).pack()
        tk.Button(root, text="Save Image", command=self.save_image).pack()

        # ---------- Universal Slider ----------
        self.slider_value = tk.IntVar(value=5)

        self.slider = tk.Scale(
            root,
            from_=1,
            to=31,
            orient=tk.HORIZONTAL,
            label="Adjustment",
            variable=self.slider_value,
            command=self.on_slider_change
        )
        self.slider.pack(pady=10)

        # ---------- Image Panels ----------
        self.panel1 = tk.Label(root)
        self.panel1.pack(side="left", padx=10)

        self.panel2 = tk.Label(root)
        self.panel2.pack(side="right", padx=10)

    # ---------- Core ----------
    def upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.bmp")]
        )
        if path:
            self.original_img = cv2.imread(path)
            self.processed_img = None
            self.show_image(self.original_img, self.panel1)

    def show_image(self, img, panel):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        panel.config(image=img_tk)
        panel.image = img_tk

    # ---------- Slider Logic ----------
    def on_slider_change(self, value):
        if self.original_img is None or self.active_filter is None:
            return

        v = int(value)

        if self.active_filter == "blur":
            if v % 2 == 0:
                v += 1
            self.processed_img = operations.gaussian_blur(self.original_img, v)

        elif self.active_filter == "sharpen":
            strength = v / 100.0
            self.processed_img = operations.laplacian_sharpen(
                self.original_img, strength
            )

        elif self.active_filter == "threshold":
            self.processed_img = operations.manual_threshold(
                self.original_img, v
            )

        elif self.active_filter == "erosion":
            self.processed_img = operations.erosion(
                self.original_img, v
            )

        self.show_image(self.processed_img, self.panel2)

    # ---------- Filter Selectors ----------
    def select_blur(self):
        self.active_filter = "blur"
        self.slider.config(label="Gaussian Blur Kernel", from_=1, to=31)
        self.slider_value.set(5)

    def select_sharpen(self):
        self.active_filter = "sharpen"
        self.slider.config(label="Sharpen Strength", from_=0, to=100)
        self.slider_value.set(50)

    def select_threshold(self):
        self.active_filter = "threshold"
        self.slider.config(label="Threshold Value", from_=0, to=255)
        self.slider_value.set(128)

    def select_erosion(self):
        self.active_filter = "erosion"
        self.slider.config(label="Erosion Iterations", from_=1, to=10)
        self.slider_value.set(1)

    # ---------- Simple Filters ----------
    def apply_grayscale(self):
        if self.original_img is None:
            return
        self.processed_img = operations.grayscale(self.original_img)
        self.show_image(
            cv2.cvtColor(self.processed_img, cv2.COLOR_GRAY2BGR),
            self.panel2
        )

    def apply_hist(self):
        if self.original_img is None:
            return
        self.processed_img = operations.histogram_equalization(self.original_img)
        self.show_image(
            cv2.cvtColor(self.processed_img, cv2.COLOR_GRAY2BGR),
            self.panel2
        )

    def save_image(self):
        if self.processed_img is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            cv2.imwrite(path, self.processed_img)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
