"""
Allows to view the calibration on the datasets.
Moving the mouse on one camera places a point on the same spot on other cameras, based on the calibration matrix.

Internal debugging utility, no real usage.
"""
import sys

import cv2

from epfl_scripts.Utilities import KEY
from epfl_scripts.Utilities.MultiCameraVisor import MultiCameraVisor
from epfl_scripts.Utilities.colorUtility import C_GREY, C_GREEN, C_BLUE, C_RED
from epfl_scripts.Utilities.geometry2D_utils import f_multiply, Point2D, Bbox, f_add, f_subtract
from epfl_scripts.Utilities.geometryCam import to3dCylinder
from epfl_scripts.groundTruthParser import getGroupedDatasets, getVideo, getCalibrationMatrix

FLOOR = '__floor__'


class Data:
    def __init__(self):
        self.valid = False

        # 2d bbox input
        self.bbox = None
        self.bbox_dataset = None

        # 3d cylinder output
        self.cylinder = None

        # 3d mouse point
        self.mouse = None

        # other 3d points
        self.points = None

    def setMouse(self, p, dataset):
        if dataset is None:
            self.mouse = p
        else:
            matrix = getCalibrationMatrix(dataset)
            self.mouse = f_multiply(matrix, p)

    def pointOnFloor(self, p):
        self.bbox_dataset = None
        self.points = None
        if self.cylinder is not None:
            self.cylinder.setCenter(p)

    def pointOnDataset(self, dataset, p):
        if self.bbox is not None:
            self.bbox_dataset = dataset
            self.bbox = Bbox.FeetWH(p, self.bbox.width, self.bbox.height)
            self.refreshCylinder()

            # draw horizontal point
            matrix = getCalibrationMatrix(dataset)
            center = self.cylinder.getCenter()
            cwidth = self.cylinder.getWidth()

            self.points = [f_add(f_subtract(f_multiply(matrix, f_add(p, Point2D(10, 0))), center).normalize(cwidth), center)]

    def refreshCylinder(self):
        self.cylinder = to3dCylinder(self.bbox_dataset, self.bbox)

    def startDrag(self, dataset, p):
        x, y = p.getAsXY()
        self.bbox = Bbox.XmYmWH(x, y, 0, 0)
        self.bbox_dataset = dataset
        self.refreshCylinder()

    def drag(self, dataset, p):
        if dataset != self.bbox_dataset:
            self.startDrag(dataset, p)
        else:
            x, y = p.getAsXY()
            self.bbox.changeXmax(x)
            self.bbox.changeYmax(y)
            self.bbox_dataset = dataset
            self.refreshCylinder()


class CalibrationParser:

    def __init__(self, groupDataset):

        # init variables
        self.Visor = MultiCameraVisor(groupDataset, "Cameras", "Floor")
        self.data = Data()
        self.groupDataset = groupDataset
        self.frames = {}

        # initialize frames
        for dataset in groupDataset:
            video = getVideo(dataset)

            # Exit if video not opened.
            if not video.isOpened():
                print("Could not open video for dataset", dataset)
                sys.exit()

            # Read first frame.
            video.set(cv2.CAP_PROP_POS_FRAMES, 1000)
            ok, frame = video.read()
            if not ok:
                print("Cannot read video file")
                sys.exit()
            self.frames[dataset] = frame
            video.release()

        self.Visor.setFrames(self.frames)

        # callback
        self.Visor.setCallback(self.clickEvent)

        # loop
        while True:

            self.updateViews()

            k = self.Visor.getKey(0)
            if k == KEY.ESC:
                break
            elif k == KEY.RIGHT_ARROW or k == KEY.D:
                pass
            elif k == KEY.LEFT_ARROW or k == KEY.A:
                pass
            elif k == KEY.UP_ARROW or k == KEY.W:
                pass
            elif k == KEY.DOWN_ARROW or k == KEY.S:
                pass

    def clickEvent(self, event, x, y, flags, dataset):
        # print event, flags

        if event == cv2.EVENT_MBUTTONUP:
            # debug
            # bboxFrom = self.data.bbox
            # cylinderMedium = to3dCylinder(dataset, bboxFrom)
            # bboxTo = from3dCylinder(dataset, cylinderMedium)
            pass

        p = Point2D(x, y)

        self.data.setMouse(p, dataset if dataset != FLOOR else None)

        if dataset == FLOOR:
            self.data.pointOnFloor(p)
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.data.startDrag(dataset, p)
            elif flags == cv2.EVENT_FLAG_LBUTTON and event != cv2.EVENT_LBUTTONUP:
                self.data.drag(dataset, p)
            else:
                self.data.pointOnDataset(dataset, p)

        self.updateViews()

    def updateViews(self):

        self.Visor.setFrames(self.frames, copy=True)

        # input bbox
        if self.data.bbox is not None and self.data.bbox_dataset is not None:
            self.Visor.drawBbox(self.data.bbox, self.data.bbox_dataset, color=C_RED, thickness=3)

        # cylinder
        if self.data.cylinder is not None:
            self.Visor.drawCylinder(self.data.cylinder, color=C_BLUE, thickness=1)

        # draw the mouse point
        if self.data.mouse is not None:
            self.Visor.drawFloorPoint(self.data.mouse, color=C_GREEN, thickness=5, drawHeadPoint=True)

        # draw other points
        if self.data.points is not None:
            for point in self.data.points:
                self.Visor.drawFloorPoint(point, color=C_GREY, thickness=5)

        self.Visor.showAll()


if __name__ == '__main__':
    CalibrationParser(getGroupedDatasets()['Laboratory/6p'])

    # list = getGroupedDatasets().values()
    # list.reverse()
    # for dataset in list:
    #     print(dataset)
    #     CalibrationParser(dataset)
