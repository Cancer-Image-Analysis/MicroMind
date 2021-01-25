import tifffile
import cv2
import numpy as np

from cania_utils.vector import Vector

""" read images """


def read_rgb(filename):
    return cv2.imread(filename, cv2.IMREAD_COLOR)


def read_gray(filename):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


def read_mirax(filename):
    pass


def read_lsm(filename):
    return tifffile.imread(filename)


def read_tiff(filename):
    return tifffile.imread(filename)


""" write images """


def write_rgb(filename, rgb_image):
    cv2.imwrite(filename, rgb_image)


def write_bgr(filename, rgb_image):
    cv2.imwrite(filename, rgb2bgr(rgb_image))


def write_tiff(filename, tiff_image, imagej=True):
    tifffile.imwrite(filename, tiff_image, imagej=imagej)


def write_gray():
    pass


""" new image """


def new_image(shape, dtype=np.uint8):
    return np.zeros(shape=shape, dtype=dtype)


""" color conversion """


def bgr2hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def hsv2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def rgb2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def gray2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


""" channels """


def split_channels(image):
    return list(cv2.split(image))


""" draw on images """


def overlay(img, mask, color=[255, 255, 0], alpha=0.4, border_color='same'):
    # Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    out = img.copy()
    img_layer = img.copy()
    img_layer[np.where(mask)] = color
    overlayed = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if border_color == 'same':
        cv2.drawContours(overlayed, contours, -1, color, 2)
    elif border_color is not None:
        cv2.drawContours(overlayed, contours, -1, border_color, 2)
    return overlayed


def fill_ellipses(mask, ellipses):
    for ellipse in ellipses:
        cv2.ellipse(mask, ellipse, 1, thickness=-1)
    return mask


""" operations """


def resize(img, scale):
    return cv2.resize(img, scale)


def count_in_mask(image, mask, threshold=0):
    _, image_th = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY)
    return np.count_nonzero(cv2.bitwise_and(image_th, image_th, mask=mask))


def mean_in_mask(image, mask):
    return np.mean(cv2.bitwise_and(image, image, mask=mask))


def split_mask_with_line(mask, line):
    line_mask = new_image(mask.shape)
    line_mask = cv2.line(line_mask, line[0], line[1], 1, 2)
    splitted_mask = cv2.bitwise_and(mask, cv2.bitwise_not(line_mask))
    contours, _ = cv2.findContours(splitted_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    submasks = []
    centroids = []
    for i, c in enumerate(contours):
        submask = new_image(mask.shape)
        cv2.drawContours(submask, contours, i, 1, 2)
        M = cv2.moments(c)
        x_centroid = round(M['m10'] / M['m00'])
        y_centroid = round(M['m01'] / M['m00'])
        submasks.append(imfill(submask))
        centroids.append(Vector(x_centroid, y_centroid))
    return submasks, centroids


def intersection_with_line(mask, line):
    line_mask = new_image(mask.shape)
    line_mask = cv2.line(line_mask, line[0], line[1], 1, 2)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask_cnt = new_image(mask.shape)
    cv2.drawContours(mask_cnt, contours, -1, 1, 2)
    intersection = cv2.bitwise_and(line_mask, mask_cnt)
    centroid = np.mean(np.argwhere(intersection), axis=0)
    return centroid


def imfill(img):
    # https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
    im_floodfill = img.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = img | im_floodfill_inv

    return im_out
