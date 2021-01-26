import tifffile
import cv2
import numpy as np

from cania_utils.geometry2D.vector2D import Vector

""" read images """


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


""" write images """


def imwrite(filename, image):
    cv2.imwrite(filename, image)


def imwrite_tiff(filename, image, imagej=True):
    tifffile.imwrite(filename, image, imagej=imagej)


""" new image """


def imnew(shape, dtype=np.uint8):
    return np.zeros(shape=shape, dtype=dtype)


""" color conversion """


def bgr2hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def hsv2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


def rgb2bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def gray2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


""" channels """


def split_channels(image):
    return list(cv2.split(image))


""" draw on images """


def overlay(image, mask, color=[255, 255, 0], alpha=0.4, border_color='same'):
    # Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    out = image.copy()
    img_layer = image.copy()
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


def resize(image, scale):
    return cv2.resize(image, scale)


def count_in_mask(image, mask, threshold=0):
    _, image_th = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY)
    return np.count_nonzero(cv2.bitwise_and(image_th, image_th, mask=mask))


def mean_in_mask(image, mask):
    return np.mean(cv2.bitwise_and(image, image, mask=mask))


def split_mask_with_line(mask, line):
    line_mask = imnew(mask.shape)
    line_mask = cv2.line(line_mask, line[0], line[1], 1, 2)
    splitted_mask = cv2.bitwise_and(mask, cv2.bitwise_not(line_mask))
    contours, _ = cv2.findContours(splitted_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    submasks = []
    centroids = []
    for i, c in enumerate(contours):
        submask = imnew(mask.shape)
        cv2.drawContours(submask, contours, i, 1, 2)
        M = cv2.moments(c)
        x_centroid = round(M['m10'] / M['m00'])
        y_centroid = round(M['m01'] / M['m00'])
        submasks.append(imfill(submask))
        centroids.append(Vector(x_centroid, y_centroid))
    return submasks, centroids


def intersection_with_line(mask, line):
    line_mask = imnew(mask.shape)
    line_mask = cv2.line(line_mask, line[0], line[1], 1, 2)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask_cnt = imnew(mask.shape)
    cv2.drawContours(mask_cnt, contours, -1, 1, 2)
    intersection = cv2.bitwise_and(line_mask, mask_cnt)
    centroid = np.mean(np.argwhere(intersection), axis=0)
    return centroid


def imfill(image):
    # https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
    im_floodfill = image.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = image.shape[:2]
    mask = imnew((h+2, w+2))

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = image | im_floodfill_inv

    return im_out
