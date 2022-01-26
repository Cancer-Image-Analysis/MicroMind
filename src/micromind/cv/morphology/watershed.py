import cv2
import numpy as np
from skimage.segmentation import watershed
import time


class WatershedTransform:
    def __init__(self, backend='skimage'):
        self._backend = backend

    def apply(self, mask, markers, image_color=None, use_distance_transform=False):
        markers[mask == 0] = 0
        markers = cv2.connectedComponents(markers, connectivity=8)[1]

        if use_distance_transform:
            mask = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5)

        if self._backend == 'opencv':
            return self._cv_transform(mask, markers, image_color=image_color)
        elif self._backend == 'skimage':
            return self._skimage_transform(mask, markers, image=image_color)

    def _cv_transform(self, mask, markers, image_color=None):
        cv2.imwrite('mask.png', mask*255)
        if image_color is None:
            image_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        signal = mask.copy()
        signal[markers != 0] = 0

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[signal > 0] = 0
        cv2.imwrite('markers.png', cv2.applyColorMap((markers*8).astype(np.uint8), cv2.COLORMAP_JET))
        labels = cv2.watershed(image_color, markers)
        labels = labels - 1
        labels[labels < 1] = 0
        return labels

    def _skimage_transform(self, mask, markers, image=None):
        if image is None:
            return watershed(255-mask, markers=markers, mask=mask > 0, watershed_line=True)
        return watershed(255-image, markers=markers, mask=mask > 0, watershed_line=True)


if __name__ == '__main__':
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[64:128, 64:128] = 255
    mask[128:192, 128:192] = 255
    cv2.imwrite('mask.png', mask)

    markers = np.zeros((256, 256), dtype=np.uint8)
    markers[70:80, 70:80] = 255
    markers[164:180, 164:180] = 255
    markers[110:120, 110:120] = 255
    markers[10:20, 10:20] = 255
    cv2.imwrite('markers.png', markers)


    t0 = time.time()
    wt_skimage = WatershedTransform(backend='skimage')
    labels = wt_skimage.apply(mask.copy(), markers)
    cv2.imwrite('labels_skimage.png', cv2.applyColorMap(labels.astype(np.uint8)*48, cv2.COLORMAP_JET))

    t1 = time.time()
    wt_opencv = WatershedTransform(backend='opencv')
    labels = wt_opencv.apply(mask.copy(), markers)
    cv2.imwrite('labels_opencv.png', cv2.applyColorMap(labels.astype(np.uint8)*48, cv2.COLORMAP_JET))

    t2 = time.time()

    print(t1-t0)
    print(t2-t1)