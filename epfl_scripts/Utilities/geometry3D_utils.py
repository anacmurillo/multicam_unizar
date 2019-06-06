"""
Functions and classes math and 3d geometry related
"""
import numpy as np

from epfl_scripts.Utilities.geometry2D_utils import f_average, Point2D


class Cylinder:
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

    def setWidth(self, width):
        self.width = width

    def setHeight(self, height):
        self.height = height


def f_averageCylinders(cylinders, weights):
    points = []
    widths = []
    heights = []

    for cylinder in cylinders:
        points.append(cylinder.getCenter())
        widths.append(cylinder.getWidth())
        heights.append(cylinder.getHeight())

    return Cylinder(f_average(points, weights), np.average(widths, weights=weights), np.average(heights, weights=weights))


