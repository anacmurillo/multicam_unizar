"""
Allows to view the calibration on the datasets.
Moving the mouse on one camera places a point on the same spot on other cameras, based on the calibration matrix.

Internal debugging utility, no real usage.
"""
import sys

import cv2
import numpy as np

from epfl_scripts.Utilities.cilinderTracker import from3dCilinder, to3dCilinder
from epfl_scripts.Utilities.geometry2D_utils import f_multiply, Point2D, f_multiplyInv, Bbox
from epfl_scripts.Utilities.geometry3D_utils import Cilinder
from epfl_scripts.groundTruthParser import getGroupedDatasets, getVideo, getCalibrationMatrix, getCalibrationMatrixFull

FLOOR = '__floor__'


def toInt(v):
    return tuple(int(round(e)) for e in v)


class MultiVisor:

    def __init__(self, groupDataset):

        # init variables
        self.frames = {FLOOR: np.zeros((512, 512, 3), np.uint8)}
        self.cilinder = Cilinder(Point2D(0, 0), 0, 0)
        self.width = 0.5
        self.height = 1.75
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

            tl = toInt(f_multiply(matrix, Point2D(0, height / 2)).getAsXY())
            tr = toInt(f_multiply(matrix, Point2D(width, height / 2)).getAsXY())
            bl = toInt(f_multiply(matrix, Point2D(0, height)).getAsXY())
            br = toInt(f_multiply(matrix, Point2D(width, height)).getAsXY())

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
            self.updateViews(draw=True)

        cv2.destroyAllWindows()

    def clickEvent(self, event, x, y, flags, dataset):
        # print x, y, dataset
        if dataset == FLOOR:
            self.cilinder = Cilinder(Point2D(x, y), self.width, self.height)
            self.updateViews(draw=True)
        else:
            bbox = Bbox.FeetWH(Point2D(x, y), self.width * 50, self.height * 50)
            self.cilinder = to3dCilinder(dataset, bbox)
            self.updateViews(draw=True, data=(dataset, bbox))

    def updateViews(self, draw=False, data=None):

        distMul = 1

        for dataset in self.groupDataset:
            frame = self.frames[dataset].copy()

            if draw:

                if data is not None and data[0] == dataset:
                    # draw event
                    l, t, r, b = data[1].getAsXmYmXMYM()
                    cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 0, 255), 3, 1)

                groundM, (headT, headP), headH, distMul = getCalibrationMatrixFull(dataset)

                point = self.cilinder.getCenter()

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

                #bbox = from3dCilinder(dataset, Cilinder(point, self.width * distMul, self.height))
                bbox = from3dCilinder(dataset, self.cilinder)
                l, t, r, b = bbox.getAsXmYmXMYM()
                cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (255, 0, 0), 1, 1)

            cv2.imshow(dataset, frame)

        if draw:
            px, py = self.cilinder.getCenter().getAsXY()
            frame = self.frames[FLOOR].copy()
            color = (255, 255, 255)
            cv2.drawMarker(frame, (int(px), int(py)), color, 1, 1, 5)
            cv2.circle(frame, (int(px), int(py)), int(self.width * distMul), color, thickness=1, lineType=8)
            print(px, py)
            cv2.imshow(FLOOR, frame)


if __name__ == '__main__':
    MultiVisor(getGroupedDatasets()['Laboratory/6p'])

    # list = getGroupedDatasets().values()
    # list.reverse()
    # for dataset in list:
    #     print(dataset)
    #     MultiVisor(dataset)
