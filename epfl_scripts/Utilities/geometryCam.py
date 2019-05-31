"""
Functions and classes math and camera-geometry related
"""
import cv2

from epfl_scripts.Utilities.colorUtility import C_GREY
from epfl_scripts.Utilities.geometry2D_utils import f_euclidian, f_multiply, f_multiplyInv, f_add, f_subtract, Point2D, Bbox
from epfl_scripts.Utilities.geometry3D_utils import Cilinder
from epfl_scripts.groundTruthParser import getCalibrationMatrixFull, getCalibrationMatrix

DIST_THRESHOLD = 20  # minimum dist to assign a detection to a prediction (in floor plane coordinates)
HEIGHT_THRESHOLD = 0.5
WIDTH_THRESHOLD = 0.5


def f_similarCilinders(a, b):
    """
    Return true iff the cilinders are similar (almost-same center, almost-same width and almost-same height)
    """
    return f_euclidian(a.getCenter(), b.getCenter()) < DIST_THRESHOLD \
           and abs(a.getWidth() - b.getWidth()) < WIDTH_THRESHOLD \
           and abs(a.getHeight() - b.getHeight()) < HEIGHT_THRESHOLD


def to3dCilinder(camera, bbox):
    """
    Converts a bbox from the specified camera image coordinates to a cilinder in floor plane
    """
    groundCalib, (headType, headCalib), headHeight, _ = getCalibrationMatrixFull(camera)

    groundP = f_multiply(groundCalib, bbox.getFeet())

    width = f_euclidian(groundP, f_multiply(groundCalib, bbox.getFeet(1))) / 2. + f_euclidian(groundP, f_multiply(groundCalib, bbox.getFeet(-1))) / 2.

    feetY = bbox.getFeet().getAsXY()[1]
    if headType == 'm':
        headY = f_multiplyInv(headCalib, groundP).getAsXY()[1]
    elif headType == 'h':
        headY = headCalib
    else:
        raise AttributeError("Unknown calibration parameter: " + headType)

    # lets assume it is lineal (it is not, but with very little difference)
    height = headHeight * (bbox.getHair().getAsXY()[1] - feetY) / (headY - feetY) if headY != feetY else 0

    return Cilinder(groundP, width, height)


def from3dCilinder(camera, cilinder):
    """
    Converts a cilinder in floor plane to a bbox from the specified camera image coordinates
    """
    groundCalib, (headType, headCalib), headHeight, _ = getCalibrationMatrixFull(camera)

    center = cilinder.getCenter()

    bottom = f_multiplyInv(groundCalib, center)

    cwidth = cilinder.getWidth()

    # width = f_euclidian(f_multiplyInv(groundCalib, f_add(center, Point2D(0, cwidth))), bottom) + f_euclidian(f_multiplyInv(groundCalib, f_add(center, Point2D(cwidth, 0))), bottom)  # this is not exactly right...but should be close enough...no, it is not

    # calculate width with an exact way, let me explain:
    # -> take the bounding box center
    # -> add an horizontal vector (any, here (10,0))
    # -> translate the point to the floor
    # -> create the vector from the cilinder center to this point (subtracting points)
    # -> normalize the vector to the width of the cilinder
    # -> add the vector the cilinder center (so that the new point is on the circle)
    # -> invtranslate to the image
    # -> measure the distance to the bbox center
    # -> multiply by 2
    width = f_euclidian(bottom, f_multiplyInv(groundCalib, f_add(f_subtract(f_multiply(groundCalib, f_add(bottom, Point2D(10, 0))), center).normalize(cwidth), center))) * 2 \
        if cwidth != 0 else 0

    if headType == 'm':
        topY = f_multiplyInv(headCalib, center).getAsXY()[1]
    elif headType == 'h':
        topY = headCalib
    else:
        raise AttributeError("Unknown calibration parameter: " + headType)

    # lets assume it is lineal (it is not, but with very little difference)
    height = (bottom.getAsXY()[1] - topY) * cilinder.getHeight() / headHeight

    return Bbox.FeetWH(bottom, width, height, heightReduced=True)


def drawVisibilityLines(frame_image, frame_floor, dataset):
    """
    Draws the visibility lines (what the camera sees) from the frame_image of the dataset to the frame_floor
    """
    height, width, _ = frame_image.shape

    matrix = getCalibrationMatrix(dataset)

    tl = f_multiply(matrix, Point2D(0, height / 2))
    tr = f_multiply(matrix, Point2D(width, height / 2))
    bl = f_multiply(matrix, Point2D(0, height))
    br = f_multiply(matrix, Point2D(width, height))

    tl = f_add(bl, f_subtract(tl, bl).multiply(4))
    tr = f_add(br, f_subtract(tr, br).multiply(4))

    tl = toInt(tl.getAsXY())
    tr = toInt(tr.getAsXY())
    bl = toInt(bl.getAsXY())
    br = toInt(br.getAsXY())

    # cv2.line(self.frames[FLOOR], tl, tr, color, 1, 1)
    cv2.line(frame_floor, tr, br, C_GREY, 1, 1)
    cv2.line(frame_floor, bl, br, C_GREY, 1, 1)
    cv2.line(frame_floor, tl, bl, C_GREY, 1, 1)


def toInt(v):
    """
    rounds to int all elements of the tuple
    """
    return tuple(int(round(e)) for e in v)


def prepareBboxForDisplay(bbox):
    """
    Returns the points for displaying purposes
    """
    l, t, r, b = bbox.getAsXmYmXMYM()
    return toInt((l, t)), toInt((r, b))


def cropBbox(bbox, frame):
    """
    Crops the bounding box for displaying purposes
    (otherwise opencv gives error if outside frame)
    """
    if bbox is None: return None

    height, width, colors = frame.shape

    # (0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows)

    return Bbox.XmYmXMYM(max(round(bbox.xmin), 0), max(round(bbox.ymin), 0), min(round(bbox.xmax), width), min(round(bbox.ymax), height))
