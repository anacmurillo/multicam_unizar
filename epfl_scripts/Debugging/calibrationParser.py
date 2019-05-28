"""
Allows to view the calibration on the datasets.
Moving the mouse on one camera places a point on the same spot on other cameras, based on the calibration matrix.

Internal debugging utility, no real usage.
"""
import sys

import cv2
import numpy as np

from epfl_scripts.Utilities.cilinderTracker import from3dCilinder, to3dCilinder
from epfl_scripts.Utilities.geometry2D_utils import f_multiply, Point2D, f_multiplyInv, Bbox, f_add, f_subtract
from epfl_scripts.groundTruthParser import getGroupedDatasets, getVideo, getCalibrationMatrix, getCalibrationMatrixFull

FLOOR = '__floor__'


def toInt(v):
    return tuple(int(round(e)) for e in v)


def prepareBboxForDisplay(bbox):
    l, t, r, b = bbox.getAsXmYmXMYM()
    return toInt((l, t)), toInt((r, b))


class Data:
    def __init__(self):
        self.valid = False

        # 2d bbox input
        self.bbox = None
        self.bbox_dataset = None

        # 3d cilinder output
        self.cilinder = None

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
        if self.cilinder is not None:
            self.cilinder.setCenter(p)

    def pointOnDataset(self, dataset, p):
        if self.bbox is not None:
            self.bbox_dataset = dataset
            self.bbox = Bbox.FeetWH(p, self.bbox.width, self.bbox.height)
            self.refreshCilinder()

            # draw horizontal point
            matrix = getCalibrationMatrix(dataset)
            center = self.cilinder.getCenter()
            cwidth = self.cilinder.getWidth()

            self.points = [f_add(f_subtract(f_multiply(matrix, f_add(p, Point2D(10, 0))), center).normalize(cwidth), center)]

    def refreshCilinder(self):
        self.cilinder = to3dCilinder(self.bbox_dataset, self.bbox)

    def startDrag(self, dataset, p):
        x, y = p.getAsXY()
        self.bbox = Bbox.XmYmWH(x, y, 0, 0)
        self.bbox_dataset = dataset
        self.refreshCilinder()

    def drag(self, dataset, p):
        if dataset != self.bbox_dataset:
            self.startDrag(dataset, p)
        else:
            x, y = p.getAsXY()
            self.bbox.changeXmax(x)
            self.bbox.changeYmax(y)
            self.bbox_dataset = dataset
            self.refreshCilinder()


class MultiVisor:

    def __init__(self, groupDataset):

        # init variables
        self.frames = {FLOOR: np.zeros((512, 512, 3), np.uint8)}
        self.data = Data()
        self.groupDataset = groupDataset

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

            # draw visible lines
            height, width, _ = frame.shape

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

            color = (100, 100, 100)

            # cv2.line(self.frames[FLOOR], tl, tr, color, 1, 1)
            cv2.line(self.frames[FLOOR], tr, br, color, 1, 1)
            cv2.line(self.frames[FLOOR], bl, br, color, 1, 1)
            cv2.line(self.frames[FLOOR], tl, bl, color, 1, 1)

            # show
            cv2.imshow(dataset, self.frames[dataset])
            cv2.setMouseCallback(dataset, self.clickEvent, dataset)
        cv2.imshow(FLOOR, self.frames[FLOOR])
        cv2.setMouseCallback(FLOOR, self.clickEvent, FLOOR)
        self.updateViews()

        while True:
            k = cv2.waitKey(0)
            if k == 27:
                break
            elif k == 83 or k == 100:  # right || d
                self.width += 0.05
            elif k == 81 or k == 97:  # left || a
                t_width = self.width - 0.05
                if t_width > 0: self.width = t_width
            elif k == 82 or k == 119:  # up || w
                self.height += 0.05
            elif k == 84 or k == 115:  # down || s
                t_height = self.height - 0.05
                if t_height > 0: self.height = t_height
            self.updateViews()

        cv2.destroyAllWindows()

    def clickEvent(self, event, x, y, flags, dataset):
        #print event, flags

        if event == cv2.EVENT_MBUTTONUP:
            # debug
            bboxFrom = self.data.bbox
            cilinderMedium = to3dCilinder(dataset, bboxFrom)
            bboxTo = from3dCilinder(dataset, cilinderMedium)
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

        for dataset in self.groupDataset:
            # each dataset
            frame = self.frames[dataset].copy()

            groundM, (headT, headP), headH, distMul = getCalibrationMatrixFull(dataset)

            if self.data.bbox is not None and self.data.bbox_dataset == dataset:
                # draw the input bbox
                lt, rb = prepareBboxForDisplay(self.data.bbox)
                cv2.rectangle(frame, lt, rb, (0, 0, 255), 3, 1)

            if self.data.cilinder is not None:
                # draw the cilinder bbox
                bbox = from3dCilinder(dataset, self.data.cilinder)
                lt, rb = prepareBboxForDisplay(bbox)
                cv2.rectangle(frame, lt, rb, (255, 0, 0), 1, 1)

            if self.data.mouse is not None:
                # draw the mouse point
                point = self.data.mouse

                # bottom point
                px, py = f_multiplyInv(groundM, point).getAsXY()
                cv2.drawMarker(frame, (int(px), int(py)), (255, 255, 255), 1, 1, 5)

                # top point
                if headT == 'm':
                    ppx, ppy = f_multiplyInv(headP, point).getAsXY()
                elif headT == 'h':
                    ppx, ppy = px, headP
                else:
                    raise AttributeError("Unknown calibration parameter: " + headT)
                cv2.drawMarker(frame, (int(ppx), int(ppy)), (200, 200, 200), 1, 1, 5)

            if self.data.points is not None:
                # draw other points
                for point in self.data.points:
                    px, py = f_multiplyInv(groundM, point).getAsXY()
                    cv2.drawMarker(frame, (int(px), int(py)), (125, 125, 125), 1, 1, 5)

            cv2.imshow(dataset, frame)

        # floor
        frame = self.frames[FLOOR].copy()
        if self.data.cilinder is not None:
            # draw cilinder
            px, py = toInt(self.data.cilinder.getCenter().getAsXY())
            point = toInt((px, py))
            color = (255, 255, 255)
            cv2.circle(frame, point, int(self.data.cilinder.width), color, thickness=1, lineType=8)

            height = toInt((px, py - self.data.cilinder.height * 25))
            cv2.line(frame, point, height, color, 1, 1)

        if self.data.mouse is not None:
            # draw mouse point
            point = toInt(self.data.mouse.getAsXY())
            color = (255, 255, 255)
            cv2.drawMarker(frame, point, color, 1, 1, 5)

        if self.data.points is not None:
            # draw other points
            for point in self.data.points:
                color = (125, 125, 125)
                point = toInt(point.getAsXY())
                cv2.drawMarker(frame, point, color, 1, 1, 5)

        cv2.imshow(FLOOR, frame)


if __name__ == '__main__':
    MultiVisor(getGroupedDatasets()['Laboratory/6p'])

    # list = getGroupedDatasets().values()
    # list.reverse()
    # for dataset in list:
    #     print(dataset)
    #     MultiVisor(dataset)
