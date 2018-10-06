"""
Docs: https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
"""
import cv2
import numpy as np


def affine_transform(img):
    rows, cols, ch = img.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (cols, rows))


def concatenate(img1, img2=None, axis=0):
    """
    axis = 0 will stack the images vertically.
    axis = 1 will stack the images horizontally.
    """
    if not img2:
        img2 = img1
    return np.concatenate((img1, img2), axis=axis)


def resize(img, factor):
    """
    Specify the image size manually with fx & fy.
    This allows the use of floats for scaling.
    """
    height, width = img.shape[:2]
    return cv2.resize(img, None, fx=factor, fy=factor)


def rotate(img, degree):
    rows, cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), degree, 1)
    return cv2.warpAffine(img, rotation_matrix, (cols, rows))


def translate(img):
    pass
