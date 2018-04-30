import math
import numpy as np


class Bbox:

    def __init__(self, xmin, xmax, width, ymin, ymax, height):
        self.xmin = xmin
        self.xmax = xmax
        self.width = width
        self.ymin = ymin
        self.ymax = ymax
        self.height = height

    @classmethod
    def XmYmXMYM(cls, xmin, ymin, xmax, ymax):
        return cls(xmin, xmax, xmax - xmin, ymin, ymax, ymax - ymin)

    @classmethod
    def XmYmWH(cls, xmin, ymin, width, height):
        return cls(xmin, xmin + width, width, ymin, ymin + height, height)

    def getAsXmYmXMYM(self):
        return self.xmin, self.ymin, self.xmax, self.ymax

    def getAsXmYmWH(self):
        return self.xmin, self.ymin, self.width, self.height


def f_iou(boxA, boxB):
    """
    IOU (Intersection over Union) of both boxes.
    :return: value in range [0,1]. 0 if disjointed bboxes, 1 if equal bboxes
    """

    boxA = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]]
    boxB = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]]

    intersection = f_area([max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])])

    union = f_area(boxA) + f_area(boxB) - intersection
    return intersection / union


def f_area(bbox):
    """
    return area of bbox
    """
    return (bbox[2] - bbox[0]) * 1. * (bbox[3] - bbox[1]) if bbox[2] > bbox[0] and bbox[3] > bbox[1] else 0.


def f_subtract(pointA, pointB):
    """
    return difference of points
    :param pointA:
    :param pointB:
    :return:
    """
    return pointA[0] - pointB[0], pointA[1] - pointB[1], 1


def f_center(bbox):
    return bbox[0] + bbox[2] / 2., bbox[1] + bbox[3] / 2.


def f_euclidian(a, b):
    """
    returns the euclidian distance between the two points
    """
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def homogeneous(p):
    return np.true_divide(p[0:2], p[2])
