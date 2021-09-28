from micromind.cv.image import intersection_with_line


class Synapse:
    def __init__(self, cell_1, cell_2):
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
