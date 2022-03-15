import time

import cv2
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage


class WatershedTransform:
    pass


class WatershedOpenCV(WatershedTransform):
    pass


class WatershedSkimage(WatershedTransform):
    def __init__(self, use_dt=False, markers_distance=20):
        self.use_dt = use_dt
        self.markers_distance = markers_distance

    def _extract_markers(self, signal):
        peak_idx = peak_local_max(signal, min_distance=self.markers_distance)
        peak_mask = np.zeros_like(signal, dtype=np.uint8)
        peak_mask[tuple(peak_idx.T)] = 1
        return peak_mask

    def apply(self, signal, markers=None, mask=None):
        if self.use_dt:
            signal = ndimage.distance_transform_edt(signal)
        if markers is None:
            markers = self._extract_markers(signal)

        markers[mask == 0] = 0
        markers = cv2.connectedComponents(markers, connectivity=8)[1]

        signal_inv = 255 - signal
        labels = watershed(signal_inv, markers=markers, mask=mask, watershed_line=True)
        return mask, markers, labels


class WatershedMarkerBased(WatershedTransform):
    def __init__(self, backend="skimage", use_dt=False, min_distance=20):
        self._backend = backend
        self.use_dt = use_dt
        self.min_distance = min_distance

    def apply(self, mask, markers=None, image_color=None):
        if self._backend == "opencv":
            return self._cv_transform(mask, markers, image_color=image_color)
        elif self._backend == "skimage":
            return self._skimage_transform(mask, markers, image=image_color)

    def _cv_transform(self, mask, markers, image_color=None):
        if self.use_dt:
            mask = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5).astype(np.uint8)
        if markers is None:
            local_max = peak_local_max(mask, indices=False, min_distance=self.min_distance).astype(np.uint8)
            markers = cv2.connectedComponents(local_max, connectivity=8)[1]

        cv2.imwrite("mask.png", mask * 255)
        if image_color is None:
            image_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        signal = mask.copy()
        signal[markers != 0] = 0

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[signal > 0] = 0
        cv2.imwrite(
            "markers.png",
            cv2.applyColorMap((markers * 8).astype(np.uint8), cv2.COLORMAP_JET),
        )
        labels = cv2.watershed(image_color, markers)
        labels = labels - 1
        labels[labels < 1] = 0
        return labels



if __name__ == "__main__":
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[64:128, 64:128] = 255
    mask[128:192, 128:192] = 255
    cv2.imwrite("mask.png", mask)

    markers = np.zeros((256, 256), dtype=np.uint8)
    markers[70:80, 70:80] = 255
    markers[164:180, 164:180] = 255
    markers[110:120, 110:120] = 255
    markers[10:20, 10:20] = 255
    cv2.imwrite("markers.png", markers)

    wt_skimage = WatershedTransform(backend="skimage", use_dt=True)
    t0 = time.time()
    mask, markers, labels = wt_skimage.apply(mask.copy(), markers=markers)
    t1 = time.time()
    cv2.imwrite(
        "labels_skimage.png",
        cv2.applyColorMap(labels.astype(np.uint8) * 48, cv2.COLORMAP_JET),
    )

    wt_opencv = WatershedTransform(backend="opencv", use_dt=True)
    t2 = time.time()
    mask, markers, labels = wt_opencv.apply(mask.copy())
    t3 = time.time()
    cv2.imwrite(
        "labels_opencv.png",
        cv2.applyColorMap(labels.astype(np.uint8) * 48, cv2.COLORMAP_JET),
    )

    print(t1 - t0)
    print(t3 - t2)
