import numpy as np


class SlideRegionData(object):
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class MicroscopeRegionData(SlideRegionData):
    def __init__(
        self, x: int, y: int, width: int, height: int, tile_x: int, tile_y: int, z: int
    ):
        super().__init__(x, y, width, height)
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.z = z


class ScannerRegionData(SlideRegionData):
    def __init__(self, x: int, y: int, width: int, height: int, level: int):
        super().__init__(x, y, width, height)
        self.level = level


class MergedChannelsRegionData(SlideRegionData):
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        channel_1: int,
        channel_2: int,
        channel_3: int,
    ):
        super().__init__(x, y, width, height)
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.channel_3 = channel_3


class SlideRegion(object):
    def __init__(self, region_data: SlideRegionData, image: np.ndarray):
        self.region_data = region_data
        self.image = image

    def get_image(self):
        return self.image

    def get_region_data(self):
        return self.region_data
