"""
Allows to view the calibration on the datasets.
Moving the mouse on one camera places a point on the same spot on other cameras, based on the calibration matrix.

Internal debugging utility, no real usage.
"""
import sys

import cv2

import epfl_scripts.sergio.Functions_DatasetLaboratory as fdl
from epfl_scripts.Utilities.MultiCameraVisor import MultiCameraVisor
from epfl_scripts.Utilities.colorUtility import C_RED, C_GREEN
from epfl_scripts.Utilities.geometry2D_utils import Point2D, Bbox
from epfl_scripts.Utilities.geometryCam import createMaskFromImage, cutImage
from epfl_scripts.groundTruthParser import getGroupedDatasets, getVideo

FLOOR = '__floor__'


class Data:
    def __init__(self):
        self.valid = False

        # 2d bbox inputs
        self.bbox = None
        self.bbox_dataset = None

    def pointOnDataset(self, dataset, p):
        if self.bbox is not None:
            self.bbox_dataset = dataset
            self.bbox = Bbox.FeetWH(p, self.bbox.width, self.bbox.height)

    def startDrag(self, dataset, p):
        x, y = p.getAsXY()
        self.bbox = Bbox.XmYmWH(x, y, 0, 0)
        self.bbox_dataset = dataset

    def drag(self, dataset, p):
        if dataset != self.bbox_dataset:
            self.startDrag(dataset, p)
        else:
            x, y = p.getAsXY()
            self.bbox.changeXmax(x)
            self.bbox.changeYmax(y)
            self.bbox_dataset = dataset

    def draw(self, visor):
        # input bbox
        if self.bbox is not None and self.bbox_dataset is not None:
            visor.drawBbox(self.bbox, self.bbox_dataset, color=C_RED, thickness=3)

    def getBbox(self):
        return self.bbox, self.bbox_dataset


class CalibrationParser:

    def __init__(self, groupDataset):

        # init variables
        self.Visor = [MultiCameraVisor(groupDataset, "CamerasA", "FloorA"), MultiCameraVisor(groupDataset, "CamerasB", "FloorB")]
        self.data = [Data(), Data()]
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

        for i in [0, 1]:
            self.Visor[i].setFrames(self.frames)

            # callback
            self.Visor[i].setCallback(lambda event, x, y, flags, dataset, li=i: self.clickEvent(li, event, x, y, flags, dataset))


        # loop
        while True:

            self.updateViews()
            k = self.Visor[0].getKey(0)
            if k == 27:
                break
            elif k == 83 or k == 100:  # right || d
                pass
            elif k == 81 or k == 97:  # left || a
                pass
            elif k == 82 or k == 119:  # up || w
                pass
            elif k == 84 or k == 115:  # down || s
                pass

    def clickEvent(self, ab, event, x, y, flags, dataset):
        # print event, flags
        # print ab

        p = Point2D(x, y)

        if dataset == FLOOR:
            pass
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.data[ab].startDrag(dataset, p)
                self.updateViews()
            elif flags == cv2.EVENT_FLAG_LBUTTON and event != cv2.EVENT_LBUTTONUP:
                self.data[ab].drag(dataset, p)
                self.updateViews()
            elif flags == cv2.EVENT_FLAG_MBUTTON:
                self.data[ab].pointOnDataset(dataset, p)
                self.updateViews()


    def updateViews(self):
        for i in (0, 1):
            self.Visor[i].setFrames(self.frames, copy=True)
            self.data[i].draw(self.Visor[i])
            self.Visor[i].showAll()

        bboxA, datasetA = self.data[0].getBbox()
        bboxB, datasetB = self.data[1].getBbox()

        if bboxA is None or bboxB is None:
            return

        # show similarity
        imageA = cutImage(self.frames[datasetA], bboxA)
        imageB = cutImage(self.frames[datasetB], bboxB)

        if imageA is None or imageB is None:
            return

        maskA = createMaskFromImage(imageA)
        maskB = createMaskFromImage(imageB)

        score, data = fdl.IstheSamePerson(imageA, maskA, imageB, maskB, False, fdl)

        print score, data
        self.Visor[0].drawText(score, self.Visor[0].FLOOR, Point2D(0, 512), size=10)

        pos = 510. / len(data)
        thresholds = {'pA_ratio':0.3, 'pB_ratio':0.3, 'pB_shape':1.5, 'pA_shape':1.5, 'dist_r':-0.05, 'dist_g':-0.05, 'dist_b':-0.05}
        for k in sorted(data):
            text = k + "=" + str(data[k])
            valid = data[k] > thresholds[k] if thresholds[k] > 0 else data[k] < -thresholds[k]
            self.Visor[1].drawText(text, self.Visor[1].FLOOR, Point2D(0, pos), size=1, color=C_GREEN if valid else C_RED)
            pos += 510. / len(data)

        # draw patches
        self.Visor[0].drawImage(cv2.bitwise_or(imageA, imageA, mask=maskA), bboxA, datasetA)
        self.Visor[1].drawImage(cv2.bitwise_or(imageB, imageB, mask=maskB), bboxB, datasetB)

        for i in (0, 1):
            self.Visor[i].showAll()


if __name__ == '__main__':
    CalibrationParser(getGroupedDatasets()['Laboratory/6p'])

    # list = getGroupedDatasets().values()
    # list.reverse()
    # for dataset in list:
    #     print(dataset)
    #     CalibrationParser(dataset)
