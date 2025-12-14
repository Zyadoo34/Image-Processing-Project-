import cv2
import numpy as np

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def histogram_equalization(img):
    gray = grayscale(img)
    return cv2.equalizeHist(gray)

def gaussian_blur(img, k=5):
    return cv2.GaussianBlur(img, (k, k), 0)

def laplacian_sharpen(img, strength=0.7):
    lap = cv2.Laplacian(img, cv2.CV_64F)
    sharp = img - strength * lap
    return np.uint8(np.clip(sharp, 0, 255))


def manual_threshold(img, thresh):
    gray = grayscale(img)
    _, th = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return th


def erosion(img, iterations=1):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(img, kernel, iterations=iterations)