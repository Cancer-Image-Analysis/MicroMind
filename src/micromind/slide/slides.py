from abc import ABC
from pathlib import Path

import numpy as np
from openslide import open_slide

from micromind.cv.conversion import bgr2hsv, rgb2bgr
from micromind.slide.exceptions import EmptySlideIdError
from micromind.slide.regions import ScannerRegionData, SlideRegion


class SlideFactory(object):
    @staticmethod
    def open_slide(slide_path: str):
        slide_file = Path(slide_path)
        print(slide_file)
        if not slide_file.is_file():
            raise FileNotFoundError

        extension = slide_file.suffix
        if extension in [".mrxs", ".tif"]:
            slide = open_slide(slide_path)
        elif extension in [".czi"]:
            slide = czifile.imread(slide_path)
        else:
            raise FileNotFoundError
        return slide


class SlideABC(ABC):
    def __init__(self, slide_path: str, slide_id: str):
        if not slide_id:
            raise EmptySlideIdError
        self.slide_id = slide_id
        self.slide = SlideFactory.open_slide(slide_path)
        self.stainings = {}

    def add_staining(self, staining):
        self.stainings[staining.name] = staining

    def clean(self):
        self.stainings = {}


class SlideIHC(SlideABC):
    def __init__(self, slide_path: str, slide_id: str):
        super().__init__(slide_path, slide_id)

    def get_region(self, x: int, y: int, width: int, height: int, level: int):
        pil_region = self.slide.read_region((x, y), level, (width, height))
        region_image = np.array(pil_region)[:, :, :3]  # remove alpha
        # region_image is RGB but BGR for cv2...
        region_data = ScannerRegionData(x, y, width, height, level)
        return SlideRegion(region_data, region_image)

    def get_stainings(self, slide_region) -> dict:
        bgr = rgb2bgr(slide_region.get_image())
        hsv = bgr2hsv(bgr)
        stainings = dict()
        stainings["bgr"] = bgr
        stainings["hsv"] = hsv
        for staining_name in self.stainings.keys():
            stainings[staining_name] = self.stainings[staining_name].get_mask(hsv)
        return stainings

    def get_masked_region(self, mask, region_image):
        return cv2.bitwise_and(region_image, region_image, mask=mask)
