from micromind.geometry.vector import Vector2
from micromind.cv.image import contours, fill_contours
import numpy as np
import cv2


class Cell2D(Vector2):
    def __init__(self, cell_name, cell_mask, cell_x, cell_y):
        super().__init__(cell_x, cell_y)
        self.name = cell_name
        self.mask = cell_mask

    @property
    def area(self):
        return cv2.countNonZero(self.mask)

    @property
    def boundary(self):
        if self.area == 0:
            return None
        return contours(self.mask)

    @property
    def min_x(self):
        return np.min(self.boundary[0], axis=0)[0, 0]

    @property
    def max_x(self):
        return np.max(self.boundary[0], axis=0)[0, 0]

    @staticmethod
    def from_mask(cell_mask, cell_name, area_range=None):
        mask = np.zeros(cell_mask.shape, dtype=np.uint8)
        cnts = contours(cell_mask)
        if len(cnts) == 1:
            cnt = cnts[0]
            if len(cnt) >= 4:
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                mask = fill_contours(mask, [cnt], color=255)

                if area_range is not None:
                    area = cv2.countNonZero(mask)
                    if area_range[0] <= area <= area_range[1]:
                        return Cell2D(cell_name, mask, cx, cy)
                    else:
                        return None
                else:
                    return Cell2D(cell_name, mask, cx, cy)      
        return None
