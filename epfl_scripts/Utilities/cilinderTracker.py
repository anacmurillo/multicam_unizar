"""
Multicamera tracker for cilinders based on detections in multiple cameras

Uses a bbox tracker for the internal tracking (should be replaced)
"""
import cv2

from deep_sort import kalman_filter
from deep_sort.detection import Detection
from deep_sort.track import Track
from epfl_scripts.Utilities.geometry2D_utils import f_multiply, f_euclidian, f_multiplyInv, f_add, Point2D, Bbox, f_intersection, f_area
from epfl_scripts.Utilities.geometry3D_utils import Cilinder, f_averageCilinders
from epfl_scripts.Utilities.kf_cilinders import KalmanFilterCilinder
from epfl_scripts.groundTruthParser import getCalibrationMatrixFull


def _cutImage(image, bbox):
    """
    Returns the path of the image under the rounder bbox in (xmin, ymin, width, height) format
    """
    bboxI = [int(round(bbox[i])) for i in range(4)]
    h, w, _ = image.shape
    bboxC = f_intersection(Bbox.XmYmWH(*bboxI), Bbox.XmYmWH(0, 0, w, h))
    if f_area(bboxC) > 0:
        return image[bboxC.ymin:bboxC.ymax+1, bboxC.xmin:bboxC.xmax+1]
    else:
        return None


def to3dCilinder(camera, bbox):
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
    groundCalib, (headType, headCalib), headHeight, _ = getCalibrationMatrixFull(camera)

    center = cilinder.getCenter()

    bottom = f_multiplyInv(groundCalib, center)

    cwidth = cilinder.getWidth()

    width = f_euclidian(f_multiplyInv(groundCalib, f_add(center, Point2D(0, cwidth))), bottom) + f_euclidian(f_multiplyInv(groundCalib, f_add(center, Point2D(cwidth, 0))), bottom)  # this is not exactly right...but should be close enough

    if headType == 'm':
        topY = f_multiplyInv(headCalib, center).getAsXY()[1]
    elif headType == 'h':
        topY = headCalib
    else:
        raise AttributeError("Unknown calibration parameter: " + headType)

    # lets assume it is lineal (it is not, but with very little difference)
    height = (bottom.getAsXY()[1] - topY) * cilinder.getHeight() / headHeight

    return Bbox.FeetWH(bottom, width, height)


class CilinderTracker:

    def __init__(self, cameras):

        self.templates = {}
        self.kf = KalmanFilterCilinder()

        self.mean = None
        self.covariance = None

        self.cameras = cameras

    def init(self, images, bboxes):

        # get existing templates and average cilinder
        cilinders = []
        weights = []
        for camera in self.cameras:
            if camera in bboxes:
                self.templates[camera] = _cutImage(images[camera], bboxes[camera].getAsXmYmWH())
                cilinders.append(to3dCilinder(camera, bboxes[camera]))
                weights.append(1)

        cilinder = f_averageCilinders(cilinders, weights)

        # get not existing templates
        for camera in self.cameras:
            if camera not in bboxes:
                self.templates[camera] = _cutImage(images[camera], from3dCilinder(camera, cilinder).getAsXmYmWH())

        # initiate tracker
        self.mean, self.covariance = self.kf.initiate(cilinder.getAsXYWH())

    def update(self, images, cilinder):

        cilinders = []
        weights = []

        for camera in self.cameras:
            # get possible new cilinders from the images
            template = self.templates[camera]

            if template is None or template.size == 0:
                continue

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cv2.matchTemplate(images[camera], template, cv2.TM_CCOEFF_NORMED))
            detbbox = Bbox.XmYmWH(*(max_loc[0:2] + template.shape[1::-1]))
            cilinders.append(to3dCilinder(camera, detbbox))  # bbox in xywh format
            weights.append(1)


        if cilinder is not None:
            cilinders.append(cilinder)
            weights.append(len(weights)+1)

        if len(cilinders) == 0:
            # no available templates, lost
            return False, None

        cilinder = f_averageCilinders(cilinders, weights)

        # update
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, cilinder.getAsXYWH())

        # predict
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)

        newcilinder = self.getCilinder()

        newbboxes = {}
        for camera in self.cameras:
            newbbox = from3dCilinder(camera, newcilinder)
            self.templates[camera] = _cutImage(images[camera], newbbox.getAsXmYmWH())
            newbboxes[camera] = newbbox

        # next step
        return True, newbboxes

    def getCilinder(self):
        return Cilinder.XYWH(*self.mean[:4].copy())

    def getBboxes(self):
        return {camera: from3dCilinder(camera, self.getCilinder()) for camera in self.cameras}
