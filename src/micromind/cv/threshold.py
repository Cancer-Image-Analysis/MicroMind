import cv2

from micromind.cv.image import BINARY_DEFAULT_VALUE


def binary_threshold(image, threshold, value=BINARY_DEFAULT_VALUE):
    return cv2.threshold(image, threshold, value, cv2.THRESH_BINARY)[1]
