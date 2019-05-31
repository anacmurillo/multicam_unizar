"""
Multicamera tracker for cilinders based on detections in multiple cameras
"""

from epfl_scripts.Utilities.geometry2D_utils import Bbox, f_intersection, f_area
from epfl_scripts.Utilities.geometry3D_utils import Cilinder, f_averageCilinders
from epfl_scripts.Utilities.geometryCam import to3dCilinder, from3dCilinder
from epfl_scripts.trackers.kf_cilinders import KalmanFilterCilinder


def _cutImage(image, bbox):
    """
    Returns the path of the image under the rounded raw bbox in (xmin, ymin, width, height) format
    """
    bboxI = [int(round(bbox[i])) for i in range(4)]
    h, w, _ = image.shape
    bboxC = f_intersection(Bbox.XmYmWH(*bboxI), Bbox.XmYmWH(0, 0, w, h))
    if f_area(bboxC) > 0:
        return image[bboxC.ymin:bboxC.ymax + 1, bboxC.xmin:bboxC.xmax + 1]
    else:
        return None


class CilinderTracker:
    """
    Implementation of the tracker
    """

    def __init__(self, cameras):
        """
        Empty tracker for the current cameras list
        """
        # self.templates = {}
        self.kf = KalmanFilterCilinder()

        self.mean = None
        self.covariance = None

        self.cameras = cameras

    def init(self, images, bboxes):
        """
        Initializes the tracker with the bboxes provided (one or more)
        """

        # get existing templates and average cilinder
        cilinders = []
        weights = []
        for camera in self.cameras:
            if camera in bboxes:
                # self.templates[camera] = _cutImage(images[camera], bboxes[camera].getAsXmYmWH())
                cilinders.append(to3dCilinder(camera, bboxes[camera]))
                weights.append(1)

        cilinder = f_averageCilinders(cilinders, weights)

        # get not existing templates
        # for camera in self.cameras:
        #    if camera not in bboxes:
        #        self.templates[camera] = _cutImage(images[camera], from3dCilinder(camera, cilinder).getAsXmYmWH())

        # initiate tracker
        self.mean, self.covariance = self.kf.initiate(cilinder.getAsXYWH())

    def update(self, images, cilinder):
        """
        Runs the prediction and update steps of the tracker with the specified cilinder as measure step
        """

        # predict
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        predict_cilinder = self.getCilinder()

        cilinders = [predict_cilinder]
        weights = [1]

        # for camera in self.cameras:
        #     # get possible new cilinders from the images
        #     template = self.templates[camera]
        #
        #     if template is None or template.size == 0:
        #         continue
        #
        #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cv2.matchTemplate(images[camera], template, cv2.TM_CCOEFF_NORMED))
        #     detbbox = Bbox.XmYmWH(*(max_loc[0:2] + template.shape[1::-1]))
        #     cilinders.append(to3dCilinder(camera, detbbox))  # bbox in xywh format
        #     weights.append(1)

        if cilinder is not None:
            cilinders.append(cilinder)
            weights.append(1)

        # compute measure
        measure_cilinder = f_averageCilinders(cilinders, weights)

        # update
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, measure_cilinder.getAsXYWH())
        newcilinder = self.getCilinder()

        # compute bboxes
        newbboxes = self.getBboxes()

        # compute lost
        lost = cilinder is None

        # return
        return not lost, newbboxes

    def getCilinder(self):
        """
        Returns the current cilinder
        :return:
        """
        return Cilinder.XYWH(*self.mean[:4].copy())

    def getBboxes(self):
        """
        Returns the list of bboxes (the cilinder translated to each camera)
        :return:
        """
        return {camera: from3dCilinder(camera, self.getCilinder()) for camera in self.cameras}
