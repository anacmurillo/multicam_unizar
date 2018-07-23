"""
Functions and classes math and geometry related
"""
import math


class Bbox:
    """
    Represents a bounding box (region) on a plane.
    Can transform between 'opposite corners' [(xmin, ymin), (xmax, ymax)] and 'size' [(xmin, ymin), height, width] coordinates
    """

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

    def isValid(self):
        return self.width >= 0 \
               and self.width == self.xmax - self.xmin \
               and self.height >= 0 \
               and self.height == self.ymax - self.ymin

    def changeXmin(self, xmin):
        self.xmin = xmin
        self.width = self.xmax - self.xmin

    def changeYmin(self, ymin):
        self.ymin = ymin
        self.height = self.ymax - self.ymin

    def changeXmax(self, xmax):
        self.xmax = xmax
        self.width = self.xmax - self.xmin

    def changeYmax(self, ymax):
        self.ymax = ymax
        self.height = self.ymax - self.ymin

    def contains(self, point, margin=1):
        x, y = point.getAsXY()
        return self.xmin - margin <= x <= self.xmax + margin and self.ymin - margin <= y <= self.ymax + margin


class Point2D:
    """
    Represents a 2d point.
    Can transform between normal (x,y) and homogeneous (x,y,s) coordinates
    """

    def __init__(self, x, y, s=1):
        self.x = x
        self.y = y
        self.s = s

    def getAsXY(self):
        return float(self.x) / self.s, float(self.y) / self.s

    def getAsXYS(self):
        return self.x, self.y, self.s


def f_iou(boxA, boxB):
    """
    IOU (Intersection over Union) of both boxes.
    :return: value in range [0,1]. 0 if disjointed bboxes, 1 if equal bboxes
    """

    intersection = f_area(f_intersection(boxA, boxB))

    union = f_area(boxA) + f_area(boxB) - intersection
    return float(intersection) / union


def f_intersection(boxA, boxB):
    """
    :return: the bbox intersection of :param boxA:  with :param boxB:
    """
    return Bbox.XmYmXMYM(max(boxA.xmin, boxB.xmin), max(boxA.ymin, boxB.ymin), min(boxA.xmax, boxB.xmax), min(boxA.ymax, boxB.ymax))


def f_area(bbox):
    """
    return area of bbox
    """
    return bbox.width * 1. * bbox.height if bbox.isValid() else 0.


def f_subtract(a, b):
    """
    return difference of points a-b
    :param a:
    :param b:
    :return:
    """
    return Point2D(b.s * a.x - a.s * b.x, b.s * a.y - a.s * b.y, a.s * b.s)


def f_center(bbox):
    """
    :return: center point of the :param bbox:
    """
    return Point2D(bbox.xmin + bbox.width / 2., bbox.ymin + bbox.height / 2.)


def f_euclidian(a, b):
    """
    returns the euclidian distance between the two points
    """
    ax, ay = a.getAsXY()
    bx, by = b.getAsXY()
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)


def f_multiply(matrix, p):
    """
    :return: result of :param matrix: * :param p: (vector) as a 2d point
    :param matrix:
    :param p:
    :return:
    """
    rows = [0, 0, 0]
    for i in range(3):
        for j in range(3):
            rows[i] += matrix[i][j] * p.getAsXYS()[j]
    return Point2D(rows[0], rows[1], rows[2])


def f_average(points, weights):
    """
    :return: average point of the list of :param points: using :param weights: (both should be same length)
    :param points:
    :param weights:
    :return:
    """
    n = len(points)
    if n == 0:
        return None

    minw = min(weights)
    weights = [x - minw + 1 for x in weights]
    multw = 1. / sum(weights)
    weights = [x * multw for x in weights]

    ax = 0
    ay = 0
    for point, weight in zip(points, weights):
        px, py = point.getAsXY()
        ax += px * weight
        ay += py * weight
    return Point2D(ax, ay)
