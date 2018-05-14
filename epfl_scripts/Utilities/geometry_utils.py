import math


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

    intersection = f_area(Bbox.XmYmXMYM(max(boxA.xmin, boxB.xmin), max(boxA.ymin, boxB.ymin), min(boxA.xmax, boxB.xmax), min(boxA.ymax, boxB.ymax)))

    union = f_area(boxA) + f_area(boxB) - intersection
    return float(intersection) / union


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
    return Point2D(bbox.xmin + bbox.width / 2., bbox.ymin + bbox.height / 2.)


def f_euclidian(a, b):
    """
    returns the euclidian distance between the two points
    """
    ax, ay = a.getAsXY()
    bx, by = b.getAsXY()
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)


def f_multiply(matrix, p):
    rows = [0, 0, 0]
    for i in range(3):
        for j in range(3):
            rows[i] += matrix[i][j] * p.getAsXYS()[j]
    return Point2D(rows[0], rows[1], rows[2])


def f_average(points):
    n = len(points)
    if n == 0:
        return None

    ax = 0
    ay = 0
    for point in points:
        px, py = point.getAsXY()
        ax += px/n
        ay += py/n
    return Point2D(ax, ay)
