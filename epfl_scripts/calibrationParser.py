import numpy as np
import sys

import cv2

from epfl_scripts.Utilities.geometry_utils import f_multiply, Point2D
from epfl_scripts.Utilities.groundTruthParser import getGroupedDatasets, getVideo, getCalibrationMatrix

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
                print "Could not open video for dataset", dataset
                sys.exit()

            # Read first frame.
            ok, frame = video.read()
            if not ok:
                print "Cannot read video file"
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
        if dataset == FLOOR:
            self.updateViews(point)
        else:
            point = f_multiply(getCalibrationMatrix(dataset), point)
        self.updateViews(point)

    def updateViews(self, point=None):
        for dataset in self.groupDataset:
            frame = self.frames[dataset].copy()

            if point is not None:
                invCalib = np.linalg.inv(getCalibrationMatrix(dataset))
                px, py = f_multiply(invCalib, point).getAsXY()
                cv2.drawMarker(frame, (int(px), int(py)), (255, 255, 255), 1, 1, 5)

            cv2.imshow(dataset, frame)
        if point is not None:
            px, py = point.getAsXY()
            frame = self.frames[FLOOR].copy()
            cv2.drawMarker(frame, (int(px), int(py)), (255, 255, 255), 1, 1, 5)
            cv2.imshow(FLOOR, frame)


if __name__ == '__main__':
    list = getGroupedDatasets().values()
    list.reverse()
    for dataset in list:
        print dataset
        MultiVisor(dataset)
