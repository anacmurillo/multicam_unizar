from epfl_scripts.Utilities.geometry2D_utils import f_average, Point2D

import numpy as np


class Cilinder:
    def __init__(self, center, width, height):
        self.center = center
        self.width = width
        self.height = height

    @classmethod
    def XYWH(cls, x, y, w, h):
        return cls(Point2D(x, y), w, h)

    def getCenter(self):
        return self.center

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getAsXYWH(self):
        x, y = self.getCenter().getAsXY()
        return x, y, self.getWidth(), self.getHeight()

    def setCenter(self, center):
        self.center = center

    def setWidth(self,width):
        self.width = width

    def setHeight(self, height):
        self.height = height


def f_averageCilinders(cilinders, weights):
    points = []
    widths = []
    heights = []

    for cilinder in cilinders:
        points.append(cilinder.getCenter())
        widths.append(cilinder.getWidth())
        heights.append(cilinder.getHeight())

    return Cilinder(f_average(points, weights), np.average(widths, weights=weights), np.average(heights, weights=weights))


