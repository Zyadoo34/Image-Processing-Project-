import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import os

import operations


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Application")

        # ---------- State ----------
        self.original_img = None
        self.processed_img = None
        self.active_filter = None
        self.pipeline = []

        # ---------- Buttons ----------
        tk.Button(root, text="Upload Image", command=self.upload_image).pack()

        tk.Button(root, text="Grayscale", command=self.select_grayscale).pack()
        tk.Button(root, text="Histogram Equalization", command=self.select_hist).pack()
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

        # ---------- Pipeline UI ----------
        tk.Label(root, text="Pipeline").pack()

        self.pipeline_list = tk.Listbox(root, height=6, width=30)
        self.pipeline_list.pack()

        tk.Button(root, text="Add to Pipeline", command=self.add_to_pipeline).pack()
        tk.Button(root, text="Apply Pipeline", command=self.apply_pipeline).pack()
        tk.Button(root, text="Clear Pipeline", command=self.clear_pipeline).pack()
        tk.Button(root, text="Batch Process Images", command=self.batch_process).pack()

        # ---------- Image Panels ----------
        self.panel1 = tk.Label(root)
        self.panel1.pack(side="left", padx=10)

        self.panel2 = tk.Label(root)
        self.panel2.pack(side="right", padx=10)

    # ---------- Image Handling ----------
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
            img = operations.gaussian_blur(self.original_img, v)

        elif self.active_filter == "sharpen":
            img = operations.laplacian_sharpen(self.original_img, v / 100.0)

        elif self.active_filter == "threshold":
            img = operations.manual_threshold(self.original_img, v)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        elif self.active_filter == "erosion":
            img = operations.erosion(self.original_img, v)

        elif self.active_filter == "grayscale":
            img = operations.grayscale(self.original_img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        elif self.active_filter == "histogram":
            img = operations.histogram_equalization(self.original_img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        else:
            return

        self.processed_img = img
        self.show_image(img, self.panel2)

    # ---------- Filter Selectors ----------
    def select_grayscale(self):
        self.active_filter = "grayscale"
        self.slider.config(label="No adjustment", from_=0, to=0)
        self.slider_value.set(0)

    def select_hist(self):
        self.active_filter = "histogram"
        self.slider.config(label="No adjustment", from_=0, to=0)
        self.slider_value.set(0)

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

    # ---------- Pipeline ----------
    def add_to_pipeline(self):
        if self.active_filter is None:
            return

        op = {
            "name": self.active_filter,
            "value": self.slider_value.get()
        }

        self.pipeline.append(op)
        self.pipeline_list.insert(
            tk.END, f"{op['name']} ({op['value']})"
        )

    def apply_pipeline(self):
        if self.original_img is None or not self.pipeline:
            return

        img = self.original_img.copy()

        for op in self.pipeline:
            name = op["name"]
            v = op["value"]

            if name == "grayscale":
                img = operations.grayscale(img)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            elif name == "histogram":
                img = operations.histogram_equalization(img)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            elif name == "blur":
                if v % 2 == 0:
                    v += 1
                img = operations.gaussian_blur(img, v)

            elif name == "sharpen":
                img = operations.laplacian_sharpen(img, v / 100.0)

            elif name == "threshold":
                img = operations.manual_threshold(img, v)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            elif name == "erosion":
                img = operations.erosion(img, v)

        self.processed_img = img
        self.show_image(img, self.panel2)

    def clear_pipeline(self):
        self.pipeline.clear()
        self.pipeline_list.delete(0, tk.END)

    # ---------- Batch Processing ----------
    def batch_process(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("Images", "*.jpg *.png *.bmp")]
        )
        if not paths or not self.pipeline:
            return

        save_dir = filedialog.askdirectory()
        if not save_dir:
            return

        for path in paths:
            img = cv2.imread(path)

            for op in self.pipeline:
                name = op["name"]
                v = op["value"]

                if name == "grayscale":
                    img = operations.grayscale(img)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                elif name == "histogram":
                    img = operations.histogram_equalization(img)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                elif name == "blur":
                    if v % 2 == 0:
                        v += 1
                    img = operations.gaussian_blur(img, v)

                elif name == "sharpen":
                    img = operations.laplacian_sharpen(img, v / 100.0)

                elif name == "threshold":
                    img = operations.manual_threshold(img, v)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                elif name == "erosion":
                    img = operations.erosion(img, v)

            filename = os.path.basename(path)
            cv2.imwrite(os.path.join(save_dir, filename), img)

    # ---------- Save ----------
    def save_image(self):
        if self.processed_img is None:
            return

        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            cv2.imwrite(path, self.processed_img)


# ---------- Run ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
