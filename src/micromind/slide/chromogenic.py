from enum import Enum
from typing import List

import cv2
import numpy as np
from cania.slides.regions import ScannerRegionData, SlideRegion
from cania.slides.slides import GenericSlide
from cania.slides.stainings import (
    ConfigurableStainingIntensityRange,
    GenericStaining,
    StainingIntensityRange,
)
from cania_utils.image import bgr2hsv, rgb2bgr


class ColorFilterHSV(object):
    def __init__(self, hsv_range: StainingIntensityRange):
        self.hsv_range = hsv_range

    def get_mask(self, hsv_image) -> np.ndarray:
        mask = cv2.inRange(
            hsv_image, self.hsv_range.get_min(), self.hsv_range.get_max()
        )
        return mask


class ConfigurableColorFilterHSV(ColorFilterHSV):
    def __init__(self):
        super(ConfigurableColorFilterHSV, self).__init__(
            ConfigurableStainingIntensityRange((0, 0, 0), (180, 255, 255))
        )


class ChromogenicStaining(GenericStaining):
    def __init__(self, color_filter: ColorFilterHSV, staining_name: str):
        super().__init__(staining_name)
        self.color_filter = color_filter

    def get_mask(self, hsv_image) -> np.ndarray:
        return self.color_filter.get_mask(hsv_image)


class CompositeStaining(GenericStaining):
    def __init__(self, color_filter: ColorFilterHSV, not_in: List, staining_name: str):
        super().__init__(staining_name)
        self.color_filter = color_filter
        self.not_in = not_in

    def get_mask(self, hsv_image) -> np.ndarray:
        main_mask = self.color_filter.get_mask(hsv_image)
        for f in self.not_in:
            main_mask = cv2.subtract(main_mask, f.get_mask(hsv_image))
        return main_mask


class ConfigurableChromogenicStaining(ChromogenicStaining):
    def __init__(self, name="configurable"):
        super().__init__(ConfigurableColorFilterHSV(), name)

    def configure(self, min_range, max_range):
        self.color_filter.hsv_range.set_min(min_range)
        self.color_filter.hsv_range.set_max(max_range)


class ConfigurableChromogenicSlide(ChromogenicSlide):
    def __init__(self, slide_path: str, slide_id: str):
        super().__init__(slide_path, slide_id)
        self.add_staining(ConfigurableChromogenicStaining())

    def configure(self, min_range, max_range):
        self.stainings["configurable"].configure(min_range, max_range)


class StainingColor(Enum):
    BLACK = StainingIntensityRange((0, 0, 0), (180, 255, 100))
    ORANGE = StainingIntensityRange((15, 120, 100), (40, 255, 255))
    PURPLE = StainingIntensityRange((110, 60, 70), (160, 255, 255))
    BLUE = StainingIntensityRange((85, 80, 120), (110, 255, 255))
    BROWN = StainingIntensityRange((0, 100, 0), (15, 255, 255))


LAMP1 = ChromogenicStaining(ColorFilterHSV(StainingColor.BLACK.value), "LAMP1")
CD107A = ChromogenicStaining(ColorFilterHSV(StainingColor.BLACK.value), "CD107a")
SOX10 = ChromogenicStaining(ColorFilterHSV(StainingColor.ORANGE.value), "Sox10")
CD8 = ChromogenicStaining(ColorFilterHSV(StainingColor.PURPLE.value), "CD8")
BLUE = ChromogenicStaining(ColorFilterHSV(StainingColor.BLUE.value), "BLUE")
MELANINE = ChromogenicStaining(ColorFilterHSV(StainingColor.BROWN.value), "Melanine")
CD107a = CompositeStaining(
    ColorFilterHSV(StainingColor.BLACK.value),
    [SOX10.color_filter, CD8.color_filter, BLUE.color_filter, MELANINE.color_filter],
    "CD107a",
)
