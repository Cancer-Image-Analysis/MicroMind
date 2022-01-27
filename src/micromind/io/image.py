import cv2
import czifile
import tifffile


def imread_color(filename):
    return cv2.imread(filename, cv2.IMREAD_COLOR)


def imread_grayscale(filename):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


def imread_mirax(filename):
    pass


def imread_lsm(filename):
    return tifffile.imread(filename)


def imread_tiff(filename):
    return tifffile.imread(filename)


def imread_czi(filename):
    return czifile.imread(filename)


def imwrite(filename, image):
    cv2.imwrite(filename, image)


def imwrite_tiff(filename, image, imagej=True):
    tifffile.imwrite(filename, image, imagej=imagej)
