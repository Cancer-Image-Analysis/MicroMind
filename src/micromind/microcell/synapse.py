import cv2

from micromind.cv.image import intersection_with_line
from micromind.microcell.cell import MicroEntity


class Synapse(MicroEntity):
    def __init__(self, cell_1, cell_2, custom_data=None):
        super().__init__(f"{cell_1.name}-{cell_2.name}", custom_data)
        self.cell_1 = cell_1
        self.cell_2 = cell_2

    @property
    def angle(self):
        return self.cell_1.angle_with_x_axis(self.cell_2)

    @property
    def distance(self):
        return self.cell_1.distance(self.cell_2)

    def front_cell_1(self):
        line = [self.cell_1.as_int_tuple(), self.cell_2.as_int_tuple()]
        return intersection_with_line(self.cell_1.mask, line)

    def crop(self, padding=5):
        synapse_mask = self.cell_1.mask | self.cell_2.mask
        x, y, w, h = cv2.boundingRect(synapse_mask)
        x = max(x - padding, 0)
        y = max(y - padding, 0)
        w = w + padding * 2
        h = h + padding * 2
        return x, y, w, h
