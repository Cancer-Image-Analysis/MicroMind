from micromind.geometry.vector import Vector2
from micromind.cv.image import contours
import cv2


class Cell2D(Vector2):
    def __init__(self, cell_name, cell_mask, cell_x, cell_y):
        super().__init__(cell_x, cell_y)
        self.name = cell_name
        self.mask = cell_mask

    @property
    def area(self):
        return cv2.countNonZero(self.mask)

    @staticmethod
    def from_mask(cell_mask, cell_name):
        cnts = contours(cell_mask)
        if len(cnts) == 1:
            cnt = cnts[0]
            if len(cnt) > 4:
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                return Cell2D(cell_name, cell_mask, cx, cy)
        return None
