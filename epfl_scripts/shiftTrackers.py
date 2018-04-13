"""
Custom implementation of Meanshift and CAMshift
base code from https://docs.opencv.org/3.4.0/db/df8/tutorial_py_meanshift.html
"""
import numpy as np

import cv2

from groundTruthParser import getVideo


class MeanShiftTracker:
    """
    Meanshift tracker implementation to use as a cv2 tracker.
    """

    def __init__(self):
        self.roi_hist = None
        self.term_crit = None
        self.track_window = None

    def init(self, frame, bbox):
        self.track_window = bbox

        roi = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        return True

    def update(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        iters, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)

        return True, self.track_window


class CAMshiftTracker:
    """
    CAMshift tracker implementation to use as a cv2 tracker.
    """

    def __init__(self):
        self.roi_hist = None
        self.term_crit = None
        self.track_window = None

    def init(self, frame, bbox):
        self.track_window = bbox

        roi = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        return True

    def update(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        ret, self.track_window = cv2.CamShift(dst, self.track_window, self.term_crit)

        pts = cv2.boxPoints(ret)
        boundrect = cv2.boundingRect(pts)
        return True, boundrect


########################### internal #################################


def evaluateMeanShift(dataset):
    cap = getVideo(dataset)  # cap = cv2.VideoCapture('slow.flv')

    # take first frame of the video
    ret, frame = cap.read()
    # setup initial location of window
    # r, h, c, w = 250, 90, 400, 125  # simply hardcoded the values

    c, r, w, h = cv2.selectROI(frame, False)  # edited

    track_window = (c, r, w, h)
    # set up the ROI for tracking
    roi = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    while (1):
        ret, frame = cap.read()
        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            # apply meanshift to get the new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            # Draw it on image
            x, y, w, h = track_window
            img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
            cv2.imshow('img2', img2)
            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
            else:
                cv2.imwrite(chr(k) + ".jpg", img2)
        else:
            break
    cv2.destroyAllWindows()
    cap.release()


def evaluateCAMshift(dataset):
    cap = getVideo(dataset)  # cv2.VideoCapture('slow.flv')

    # take first frame of the video
    ret, frame = cap.read()
    # setup initial location of window
    # r, h, c, w = 250, 90, 400, 125  # simply hardcoded the values

    c, r, w, h = cv2.selectROI(frame, False)  # edited

    track_window = (c, r, w, h)
    # set up the ROI for tracking
    roi = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    while (1):
        ret, frame = cap.read()
        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            # apply meanshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame, [pts], True, 255, 2)
            cv2.imshow('img2', img2)
            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
            else:
                cv2.imwrite(chr(k) + ".jpg", img2)
        else:
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    name = "Basketball/match5-c0"
    name = "Campus/campus4-c0"
    name = "Laboratory/6p-c0"

    # evaluateMeanShift(name)
    evaluateCAMshift(name)
