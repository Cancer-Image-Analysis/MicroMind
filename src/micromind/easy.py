import cv2
import numpy as np

from micromind.cv.image import contours, fill_contours
from micromind.microcell.cell import Particle2D


def __create_entity(mask):
    mask_bin = np.zeros_like(mask, dtype=np.uint8)
    cnts = contours(mask)
    if len(cnts) != 1:
        raise ValueError(f"Can't create entity with ambiguous mask!")
    cnt = cnts[0]
    if len(cnt) < 4:
        raise ValueError(f"Can't create entity with less than 4 points!")
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        raise ValueError(f"Can't create entity with M['m00'] == 0")
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    mask_bin = fill_contours(mask_bin, [cnt], color=255)
    return mask_bin, cx, cy


def create_particle(name, mask, custom_data=None):
    mask_bin, x, y = __create_entity(mask)
    return Particle2D(name, mask_bin, x, y, custom_data=custom_data)
