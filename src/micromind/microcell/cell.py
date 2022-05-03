import math

import cv2
import numpy as np

from micromind.cv.image import contours, fill_contours, split_mask_with_lines
from micromind.geometry.line import Line2
from micromind.geometry.vector import Vector2


class MicroEntity:
    def __init__(self, name, custom_data=None):
        self.name = name
        self.custom_data = custom_data
        if self.custom_data is None:
            self.custom_data = {}

    def set_data(self, data_name, data):
        self.custom_data[data_name] = data

    def get_data(self, data_name):
        return self.custom_data[data_name]


class MicroEntity2D(Vector2, MicroEntity):
    def __init__(self, name, mask, x, y, custom_data=None):
        MicroEntity.__init__(self, name, custom_data=custom_data)
        Vector2.__init__(self, x, y)
        self.mask = mask

    def get_mean(self, channel):
        return np.mean(channel, where=self.mask > 0)

    def get_sum(self, channel):
        return np.sum(channel, where=self.mask > 0)

    def get_max(self, channel):
        return np.max(channel, where=self.mask > 0, initial=0)

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
        return 4 * math.pi * (self.area / self.perimeter ** 2)

    @property
    def min_x(self):
        return np.min(self.boundary[0], axis=0)[0, 0]

    @property
    def max_x(self):
        return np.max(self.boundary[0], axis=0)[0, 0]


class Particle2D(MicroEntity2D):
    pass


class Cell2D(MicroEntity2D):
    def __init__(self, name, mask, x, y, custom_data=None):
        super().__init__(name, mask, x, y, custom_data)

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


HOURGLASS_ORIENTATION = 45
HOURGLASS_ANGLES = np.array([0, 90, 180, 270]) - HOURGLASS_ORIENTATION


class CellHourglass(Cell2D):
    def __init__(self, cell_name, cell_mask, x, y, angle, custom_data={}):
        super().__init__(cell_name, cell_mask, x, y, custom_data)
        angles = np.radians(angle - HOURGLASS_ANGLES)

        coss = np.cos(angles)
        sins = np.sin(angles)

        A = self + Vector2(coss[0], sins[0]) * 500
        B = self + Vector2(coss[1], sins[1]) * 500
        C = self + Vector2(coss[2], sins[2]) * 500
        D = self + Vector2(coss[3], sins[3]) * 500
        front_point = (A + B) * 0.5

        AC = Line2(A, C)
        BD = Line2(B, D)

        submasks, centroids = split_mask_with_lines(self.mask, [AC, BD])
        distances = np.array([c.distance(front_point) for c in centroids])
        ordered_indices = np.argsort(distances)

        self.front = submasks[ordered_indices[0]]
        self.side1 = submasks[ordered_indices[1]]
        self.side2 = submasks[ordered_indices[2]]
        self.rear = submasks[ordered_indices[3]]
