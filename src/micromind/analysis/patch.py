from abc import ABC, abstractmethod

import numpy as np

from micromind.cv.conversion import rgb2bgr


class PatchGenerator(ABC):
    def __init__(self, patch_size):
        self._patch_size = patch_size

    @abstractmethod
    def get_patches(self, slide, all_indices):
        pass


class PatchPaddingIHC(object):
    def __init__(self, stainings, real_x, real_y, size, w, h):
        self.stainings = stainings
        self.size = size
        self.real_x = real_x
        self.real_y = real_y
        self.w = w
        self.h = h

    def get_real_surface_mask(self):
        real_surface_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        real_surface_mask[
            self.real_y : self.real_y + self.size, self.real_x : self.real_x + self.size
        ] = 1
        return real_surface_mask


class PatchGeneratorIHC(PatchGenerator):
    def __init__(self, patch_size, patch_level):
        super(PatchGeneratorIHC, self).__init__(patch_size)
        self._patch_level = patch_level

    def get_patches(self, ihc_slide, all_indices):
        # pad if possible
        for i in range(len(all_indices)):
            x = all_indices[i][1]
            y = all_indices[i][0]
            yield ihc_slide.get_region(
                x, y, self._patch_size, self._patch_size, self._patch_level
            )


class PatchGeneratorWithPaddingIHC(PatchGeneratorIHC):
    """
    add padding around patch to avoid the analysis being distorted by the edge
    of the image usefull when doing image processing
    """

    def __init__(self, size, level, padding):
        super(PatchGeneratorWithPaddingIHC, self).__init__(size, level)
        self._padding = padding

    def get_patches(self, ihc_slide, all_indices):
        min_all_indices = all_indices - self._padding
        max_all_indices = all_indices + self._padding + self._patch_size
        w_max, h_max = ihc_slide.slide.level_dimensions[self._patch_level]
        # pad if possible
        for i in range(len(min_all_indices)):
            x_min = max(0, min_all_indices[i][1])
            y_min = max(0, min_all_indices[i][0])
            x_max = min(w_max, max_all_indices[i][1])
            y_max = min(h_max, max_all_indices[i][0])
            w = x_max - x_min
            h = y_max - y_min
            real_x = all_indices[i][1] - x_min
            real_y = all_indices[i][0] - y_min
            patch_region = ihc_slide.get_region(x_min, y_min, w, h, self._patch_level)
            patch_stainings = ihc_slide.get_stainings(patch_region)

            yield x_min, y_min, PatchPaddingIHC(
                patch_stainings, real_x, real_y, self._patch_size, w, h
            )


class GridPatchAnalysis(ABC):
    def __init__(self, patch_generator, patch_analysis):
        self.patch_generator = patch_generator
        self.patch_analysis = patch_analysis

    def run(self, ihc_slide, all_indices, disk_location=None):
        results = []  # row list
        for x, y, patch in self.patch_generator.get_patches(ihc_slide, all_indices):
            result = self.patch_analysis.run(patch, id=f"{x}_{y}")
            result["patch.x"] = x
            result["patch.y"] = y
            result["patch.w"] = patch.w
            result["patch.h"] = patch.h
            result["patch.pad_x"] = patch.real_x
            result["patch.pad_y"] = patch.real_y
            results.append(result)
            if disk_location:
                disk_location.write(f"patch_{x}_{y}.png", patch.stainings["bgr"])
        return results
