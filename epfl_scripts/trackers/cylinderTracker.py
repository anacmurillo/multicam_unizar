"""
Multicamera tracker for cylinders based on detections in multiple cameras
"""

from epfl_scripts.Utilities.geometry3D_utils import Cylinder, f_averageCylinders
from epfl_scripts.Utilities.geometryCam import to3dCylinder, from3dCylinder
from epfl_scripts.trackers.kf_cylinders import KalmanFilterCylinder


class CylinderTracker:
    """
    Implementation of the tracker
    """

    def __init__(self, cameras):
        """
        Empty tracker for the current cameras list
        """
        # self.templates = {}
        self.kf = KalmanFilterCylinder()

        self.mean = None
        self.covariance = None

        self.cameras = cameras

    def init(self, images, bboxes):
        """
        Initializes the tracker with the bboxes provided (one or more)
        """

        # get existing templates and average cylinder
        cylinders = []
        weights = []
        for camera in self.cameras:
            if camera in bboxes:
                # self.templates[camera] = cutImage(images[camera], bboxes[camera])
                cylinders.append(to3dCylinder(camera, bboxes[camera]))
                weights.append(1)

        cylinder = f_averageCylinders(cylinders, weights)

        # get not existing templates
        # for camera in self.cameras:
        #    if camera not in bboxes:
        #        self.templates[camera] = cutImage(images[camera], from3dCylinder(camera, cylinder))

        # initiate tracker
        self.mean, self.covariance = self.kf.initiate(cylinder.getAsXYWH())

    def update(self, images, cylinder):
        """
        Runs the prediction and update steps of the tracker with the specified cylinder as measure step
        """

        # predict
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        predict_cylinder = self.getCylinder()

        cylinders = [predict_cylinder]
        weights = [1]

        # for camera in self.cameras:
        #     # get possible new cylinders from the images
        #     template = self.templates[camera]
        #
        #     if template is None or template.size == 0:
        #         continue
        #
        #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cv2.matchTemplate(images[camera], template, cv2.TM_CCOEFF_NORMED))
        #     detbbox = Bbox.XmYmWH(*(max_loc[0:2] + template.shape[1::-1]))
        #     cylinders.append(to3dCylinder(camera, detbbox))  # bbox in xywh format
        #     weights.append(1)

        if cylinder is not None:
            cylinders.append(cylinder)
            weights.append(1)

        # compute measure
        measure_cylinder = f_averageCylinders(cylinders, weights)

        # update
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, measure_cylinder.getAsXYWH())
        newcylinder = self.getCylinder()

        # compute bboxes
        newbboxes = self.getBboxes()

        # compute lost
        lost = cylinder is None

        # return
        return not lost, newbboxes

    def getCylinder(self):
        """
        Returns the current cylinder
        :return:
        """
        return Cylinder.XYWH(*self.mean[:4].copy())

    def getBboxes(self):
        """
        Returns the list of bboxes (the cylinder translated to each camera)
        :return:
        """
        return {camera: from3dCylinder(camera, self.getCylinder()) for camera in self.cameras}
