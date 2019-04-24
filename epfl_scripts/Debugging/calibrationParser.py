"""
Allows to view the calibration on the datasets.
Moving the mouse on one camera places a point on the same spot on other cameras, based on the calibration matrix.

Internal debugging utility, no real usage.
"""
import sys

import cv2
import numpy as np

from epfl_scripts.Utilities.geometry2D_utils import f_multiply, Point2D, f_multiplyInv
from epfl_scripts.Utilities.geometry3D_utils import Cilinder
from epfl_scripts.groundTruthParser import getGroupedDatasets, getVideo, getCalibrationMatrix, getCalibrationMatrixFull
from epfl_scripts.multiCameraTrackerV2 import from3dCilinder

FLOOR = '__floor__'


class MultiVisor:
    frames = {FLOOR: np.zeros((512, 512, 3), np.uint8)}
    x = 0
    y = 0
    groupDataset = None

    def __init__(self, groupDataset):
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
            cv2.imshow(dataset, self.frames[dataset])
            cv2.setMouseCallback(dataset, self.clickEvent, dataset)
        cv2.imshow(FLOOR, self.frames[FLOOR])
        cv2.setMouseCallback(FLOOR, self.clickEvent, FLOOR)
        self.updateViews()
        while cv2.waitKey(0) != 27:
            pass
        cv2.destroyAllWindows()

    def clickEvent(self, event, x, y, flags, dataset):
        # print x, y, dataset
        point = Point2D(x, y)
        if dataset != FLOOR:
            point = f_multiply(getCalibrationMatrix(dataset), point)
        self.updateViews(point)

    def updateViews(self, point=None):

        for dataset in self.groupDataset:
            frame = self.frames[dataset].copy()

            if point is not None:
                groundM, (headT, headP), headH = getCalibrationMatrixFull(dataset)

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

                bbox = from3dCilinder(dataset, Cilinder(point, 0.5 * 50, 1.75))  # 50=magic number, found by testing
                l, t, r, b = bbox.getAsXmYmXMYM()
                cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (255, 0, 0), 1, 1)

            cv2.imshow(dataset, frame)
        if point is not None:
            px, py = point.getAsXY()
            frame = self.frames[FLOOR].copy()
            cv2.drawMarker(frame, (int(px), int(py)), (255, 255, 255), 1, 1, 5)
            cv2.imshow(FLOOR, frame)


if __name__ == '__main__':
    #MultiVisor(getGroupedDatasets()['Laboratory/6p'])

    list = getGroupedDatasets().values()
    list.reverse()
    for dataset in list:
        print(dataset)
        MultiVisor(dataset)
