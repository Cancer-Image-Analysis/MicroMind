import matplotlib.pyplot as plt
from matplotlib import rc

from micromind.cv.image import contours, draw_contours


rc('font', **{'size': 6})


def plot_predictions(original, prediction, contours_color=[255, 255, 0]):
    fig = plt.figure(figsize=(6, 18), dpi=300)

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(original)
    ax.set_title("original image")
    ax.axis('off')

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(prediction, cmap="viridis")
    ax.set_title("prediction")
    ax.axis('off')

    borders = original.copy()
    cnts = contours(prediction)
    borders = draw_contours(borders, cnts, color=contours_color)
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(borders)
    ax.set_title("contours")
    ax.axis('off')
