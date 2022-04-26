import cv2
import numpy as np

from micromind.geometry.vector import Vector2

BINARY_DEFAULT_VALUE = 255


def imnew(shape, dtype=np.uint8):
    return np.zeros(shape=shape, dtype=dtype)


def split_channels(image):
    return list(cv2.split(image))


def contours(image, exclude_holes=True):
    mode = cv2.RETR_LIST
    method = cv2.CHAIN_APPROX_SIMPLE
    if exclude_holes:
        mode = cv2.RETR_EXTERNAL
    return cv2.findContours(image.copy(), mode, method)[0]


def fill_contours(image, cnts, color=[255, 255, 0], cnt_index=-1):
    return cv2.drawContours(image.copy(), cnts, cnt_index, color, -1)


def draw_contours(image, cnts, thickness=2, color=[0, 255, 255], cnt_index=-1):
    return cv2.drawContours(image, cnts, cnt_index, color, thickness)


def imfill(image):
    cnts = contours(image, exclude_holes=True)
    return fill_contours(image, cnts, color=BINARY_DEFAULT_VALUE)


def overlay(image, mask, color=[255, 255, 0], alpha=0.4, border_color="same"):
    out = image.copy()
    img_layer = image.copy()
    img_layer[np.where(mask)] = color
    overlayed = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
    cnts = contours(mask, exclude_holes=True)
    if border_color == "same":
        draw_contours(overlayed, cnts, thickness=1, color=color)
    elif border_color is not None:
        draw_contours(overlayed, cnts, thickness=1, color=border_color)
    return overlayed


def fill_ellipses(mask, ellipses, color=BINARY_DEFAULT_VALUE):
    for ellipse in ellipses:
        cv2.ellipse(mask, ellipse, color, thickness=-1)
    return mask


def fill_ellipses_as_labels(mask, ellipses):
    for i, ellipse in enumerate(ellipses):
        cv2.ellipse(mask, ellipse, i + 1, thickness=-1)
    return mask


def fill_polygons(mask, polygons, color=BINARY_DEFAULT_VALUE):
    return cv2.fillPoly(mask, pts=polygons, color=color)


def fill_polygons_as_labels(mask, polygons):
    for i, polygon in enumerate(polygons):
        cv2.fillPoly(mask, pts=np.int32([polygon]), color=i + 1)
    return mask


""" operations """


def resize(image, scale):
    return cv2.resize(image, scale)


def count_in_mask(image, mask, threshold=0):
    _, image_th = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY)
    return np.count_nonzero(cv2.bitwise_and(image_th, image_th, mask=mask))


def max_in_mask(image, mask):
    image_on_mask = cv2.bitwise_and(image, image, mask=mask)
    only_positive_values = image_on_mask[np.argwhere(image_on_mask)]
    if len(only_positive_values) == 0:
        return 0.0
    return np.max(only_positive_values)


def mean_value(image, mask=None, threshold=None):
    if mask is None and threshold is None:
        return np.mean(image)
    if mask is not None and threshold is None:
        return np.mean(image, where=mask > 0)
    if mask is None and threshold is not None:
        mask = image >= threshold
        if np.all(mask == 0):
            return None
        return np.mean(image, where=mask)
    mask[image < threshold] = 0
    return np.mean(image, where=mask > 0)


def split_mask_with_lines(mask, lines):
    line_mask = imnew(mask.shape)
    for line in lines:
        line_mask = cv2.line(
            line_mask,
            line.pt1.as_int_tuple(),
            line.pt2.as_int_tuple(),
            BINARY_DEFAULT_VALUE,
            2,
        )
    splitted_mask = cv2.bitwise_and(mask, cv2.bitwise_not(line_mask))
    cnts = contours(splitted_mask)
    submasks = []
    centroids = []
    for i, c in enumerate(cnts):
        submask = imnew(mask.shape)
        cv2.drawContours(submask, cnts, i, 1, 2)
        M = cv2.moments(c)
        if M["m00"] == 0:
            x_centroid = 0
            y_centroid = 0
        else:
            x_centroid = round(M["m10"] / M["m00"])
            y_centroid = round(M["m01"] / M["m00"])
        submasks.append(imfill(submask))
        centroids.append(Vector2(x_centroid, y_centroid))
    return submasks, centroids


def intersection_with_line(mask, line):
    line_mask = imnew(mask.shape)
    line_mask = cv2.line(line_mask, line[0], line[1], BINARY_DEFAULT_VALUE, 2)
    cnts = contours(mask)
    mask_cnt = imnew(mask.shape)
    cv2.drawContours(mask_cnt, cnts, -1, BINARY_DEFAULT_VALUE, 2)
    intersection = cv2.bitwise_and(line_mask, mask_cnt)
    centroid = np.mean(np.argwhere(intersection), axis=0)
    return centroid


def mean_over_line(image, line, thickness=2):
    line_mask = imnew(image.shape)
    line_mask = cv2.line(line_mask, line[0], line[1], 255, thickness)
    return mean_value(image, mask=line_mask)


def max_over_line(image, line):
    line_mask = imnew(image.shape)
    line_mask = cv2.line(line_mask, line[0], line[1], BINARY_DEFAULT_VALUE, 2)
    return max_in_mask(image, line_mask)


def extract_rectangle_area(image, center, theta, width, height, flags=cv2.INTER_LINEAR):
    """
    Rotates OpenCV image around center with angle theta (in deg)
    then crops the image according to width and height.
    """
    shape = (image.shape[1], image.shape[0])

    matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
    image = cv2.warpAffine(src=image, M=matrix, dsize=shape, flags=flags)

    x = max(0, int(center[0] - width / 2))
    y = max(0, int(center[1] - height / 2))
    image = image[y : y + height, x : x + width]

    return image
