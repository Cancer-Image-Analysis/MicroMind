from abc import ABC

import cv2


class StainingColor:
    def __init__(self, hsv_min, hsv_max):
        self.hsv_min = hsv_min
        self.hsv_max = hsv_max

    @property
    def min(self):
        return self.hsv_min

    @property
    def max(self):
        return self.hsv_max


BLACK = StainingColor((0, 0, 0), (180, 255, 100))
ORANGE = StainingColor((15, 120, 100), (40, 255, 255))
PURPLE = StainingColor((110, 60, 70), (160, 255, 255))
BLUE = StainingColor((85, 80, 120), (110, 255, 255))
BROWN = StainingColor((0, 100, 0), (15, 255, 255))


class StainingABC(ABC):
    def __init__(self, name: str):
        self.name = name


class StainingIHC(StainingABC):
    def __init__(self, name: str, hsv_min, hsv_max):
        super().__init__(name)
        self.hsv_min = hsv_min
        self.hsv_max = hsv_max

    def get_mask(self, image):
        return cv2.inRange(image, self.hsv_min, self.hsv_max)


CD107A = StainingIHC("CD107a", BLACK.min, BLACK.max)
SOX10 = StainingIHC("Sox10", ORANGE.min, ORANGE.max)
CD8 = StainingIHC("CD8", PURPLE.min, PURPLE.max)
DAPI = StainingIHC("BLUE", BLUE.min, BLUE.max)


class ConfocalStaining(StainingABC):
    def __init__(self, name: str, channel: int, threshold=0):
        super().__init__(name)
        self.channel = channel
        self.threshold = threshold


class StainingIntensityRange(object):
    def __init__(self, min_range: tuple, max_range: tuple):
        self._min = min_range
        self._max = max_range

    def get_min(self):
        return self._min

    def get_max(self):
        return self._max


class ConfigurableStainingIntensityRange(StainingIntensityRange):
    def __init__(self, min_range: tuple, max_range: tuple):
        super().__init__(min_range, max_range)

    def set_min(self, min_range: tuple):
        self._min = min_range

    def set_max(self, max_range: tuple):
        self._max = max_range


# FM4_64 = ConfocalStaining('FM4-64')
# PKH_67 = ConfocalStaining('PKH-67')
# TUBULIN = ConfocalStaining('Tubulin')
# CASPASE_3 = ConfocalStaining('Caspase-3')
