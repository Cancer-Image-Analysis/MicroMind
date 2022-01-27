from dataclasses import dataclass
from math import pi
from typing import List

import matplotlib.pyplot as plt


@dataclass
class Figure:
    title: str = None
    size: tuple = (12, 12)
    dpi: int = 300
    background_color: tuple = (1, 1, 1)
    hspace: float = 0.25

    def create_panels(self, nrows=2, ncols=2):
        self.__fig, self.panels = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=self.size, dpi=self.dpi
        )
        if self.title:
            self.__fig.suptitle(self.title)
        self.__fig.set_facecolor(self.background_color)
        self.__fig.subplots_adjust(hspace=self.hspace)
        self.panels = self.panels.ravel()

    def get_panel(self, n):
        return self.panels[n]

    def set_projection(self, n, projection):
        self.panels[n].remove()
        self.panels[n] = self.__fig.add_subplot(3, 2, n + 1, projection=projection)

    def save(self, filename):
        self.__fig.savefig(filename)


@dataclass
class FigureSinglePanel:
    title: str = None
    xlabel: str = None
    ylabel: str = None
    size: tuple = (12, 12)
    dpi: int = 300
    background_color: tuple = (1, 1, 1)

    def __post_init__(self):
        self.__fig, self.panel = plt.subplots(
            nrows=1, ncols=1, figsize=self.size, dpi=self.dpi
        )
        if self.title:
            self.__fig.suptitle(self.title)
        if self.xlabel:
            self.panel.set_xlabel(self.xlabel)
        if self.ylabel:
            self.panel.set_ylabel(self.ylabel)
        self.__fig.set_facecolor(self.background_color)

    def get_panel(self):
        return self.panel

    def set_projection(self, n, projection):
        self.panels[n].remove()
        self.panels[n] = self.__fig.add_subplot(3, 2, n + 1, projection=projection)

    def save(self, filename):
        self.__fig.savefig(filename)


@dataclass
class FigureSingleImage(FigureSinglePanel):
    def __post_init__(self):
        super().__post_init__()
        self.panel.axis("off")

    def get_panel(self):
        return self.panel

    def set_projection(self, n, projection):
        self.panels[n].remove()
        self.panels[n] = self.__fig.add_subplot(3, 2, n + 1, projection=projection)

    def save(self, filename):
        self.__fig.savefig(filename)


@dataclass
class FigureSinglePolar(FigureSinglePanel):
    def __post_init__(self):
        self.__fig, self.panel = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=self.size,
            dpi=self.dpi,
            subplot_kw={"projection": "polar"},
        )
        N = len(self.xlabel)
        # If you want the first axis to be on top:
        self.panel.set_theta_offset(pi / 2)
        self.panel.set_theta_direction(-1)

        self.angles = [n / float(N) * 2 * pi for n in range(N)]
        self.angles += self.angles[:1]

        # Draw ylabels
        self.panel.set_rlabel_position(0)
        self.panel.set_rticks([0.25, 0.5, 0.75])  # Less radial ticks
        self.panel.set_rmax(1)
        if self.title:
            self.__fig.suptitle(self.title)
        if self.xlabel:
            self.panel.set_xticks(self.angles[:-1], minor=False)
            self.panel.set_xticklabels(self.xlabel, fontdict=None, minor=False)
        self.panel.plot(self.angles, [0] * (N + 1), linewidth=0)
        self.__fig.set_facecolor(self.background_color)

    def get_panel(self):
        return self.panel

    def save(self, filename):
        self.__fig.savefig(filename)
