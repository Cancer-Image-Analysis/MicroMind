import cv2
import numpy as np

from micromind.cv.image import contours, fill_contours
from micromind.geometry.vector import Vector2
import math


class MicroObject:
    def __init__(self, custom_data={}):
        self.custom_data = custom_data

    def set_data(self, data_name, data):
        self.custom_data[data_name] = data

    def get_data(self, data_name):
        return self.custom_data[data_name]


class Cell2D(Vector2, MicroObject):
    def __init__(self, cell_name, cell_mask, x, y, custom_data={}):
        Vector2.__init__(self, x, y)
        MicroObject.__init__(self, custom_data=custom_data)
        self.name = cell_name
        self.mask = cell_mask

    def get_mean(self, channel):
        return np.mean(channel, where=self.mask > 0)

    @property
    def area(self):
        return cv2.countNonZero(self.mask)

    @property
    def boundary(self):
        if self.area == 0:
            return None
        return contours(self.mask)

    @property
    def perimeter(self):
        return cv2.arcLength(self.boundary[0], True)

    @property
    def roundness(self):
        return 4 * math.pi * (self.area / self.perimeter**2)

    @property
    def min_x(self):
        return np.min(self.boundary[0], axis=0)[0, 0]

    @property
    def max_x(self):
        return np.max(self.boundary[0], axis=0)[0, 0]

    @staticmethod
    def from_mask(cell_mask, cell_name, area_range=None, custom_data={}):
        mask = np.zeros(cell_mask.shape, dtype=np.uint8)
        cnts = contours(cell_mask)
        if len(cnts) == 1:
            cnt = cnts[0]
            if len(cnt) >= 4:
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    return None
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                mask = fill_contours(mask, [cnt], color=255)

                if area_range is not None:
                    area = cv2.countNonZero(mask)
                    if area_range[0] <= area <= area_range[1]:
                        return Cell2D(cell_name, mask, cx, cy, custom_data=custom_data)
                    else:
                        return None
                else:
                    return Cell2D(cell_name, mask, cx, cy, custom_data=custom_data)
        return None
