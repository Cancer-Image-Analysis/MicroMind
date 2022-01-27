import numpy as np

from micromind.cv.image import mean_over_line, mean_value


class TestMeanValue:
    def test_line(self):
        w = 5
        h = 5
        image = np.zeros((h, w), dtype=np.uint8)
        image[2, 1:4] = 1
        line = [[0, 0], [0, 4]]
        res = mean_over_line(image, line, thickness=1)
        assert res == 0

        line = [[2, 0], [2, 4]]
        res = mean_over_line(image, line, thickness=1)
        assert res == 3 / 15

        res = mean_over_line(image, line, thickness=3)
        assert res == 3 / 25

    def test_zero(self):
        w = 5
        h = 5
        image = np.zeros((h, w), dtype=np.uint8)
        res = mean_value(image)
        assert res == 0

        image[2, 2] = 1
        res = mean_value(image)
        assert res == 0.04

        image[2, 2] = 25
        res = mean_value(image)
        assert res == 1

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[2, 2] = True
        res = mean_value(image, mask)
        assert res == 25

        mask[1:4, 2] = True
        mask[2, 1:4] = True
        res = mean_value(image, mask)
        assert res == 5

        image[2, 2] = 0
        res = mean_value(image, mask)
        assert res == 0

        image[1:4, 2] = 1
        image[2, 1:4] = 1
        res = mean_value(image, mask)
        assert res == 1

        image[2, 2] = 2
        res = mean_value(image, mask)
        assert res == 1.2

        res = mean_value(image, threshold=2)
        assert res == 2

        res = mean_value(image, mask, threshold=2)
        assert res == 2

        res = mean_value(image, threshold=3)
        assert res is None
