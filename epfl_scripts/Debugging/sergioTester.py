"""
Allows to view the Functions_DatasetLaboratory similarity between two functions dinamically
Use Q,W,E,A,S,D,0,1,2,3,4,5,6,7,8,9,START,END to advance/rewind the videos
Draw from top-left to bottom-right corner to create rectangles
Middle-mouse press and drag to move the rectangle
F5 to reload the module

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


def reloadModule():
    # reloads the module
    reload(fdl)


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

    def draw(self, visor, color):
        # input bbox
        if self.bbox is not None and self.bbox_dataset is not None:
            visor.drawBbox(self.bbox, self.bbox_dataset, color=color, thickness=3)

    def getBbox(self):
        return self.bbox, self.bbox_dataset


class CalibrationParser:

    def __init__(self, groupDataset):

        # init variables
        self.Visor = [MultiCameraVisor(groupDataset, "CamerasA", "FloorA"), MultiCameraVisor(groupDataset, "CamerasB", "FloorB")]
        self.data = [Data(), Data()]
        self.groupDataset = groupDataset
        self.frames = [{}, {}]
        self.indexes = [0, 0]
        self.length = 0
        self.videos = {}
        self.lastdataset = 0

        # initialize videos
        for dataset in groupDataset:
            video = getVideo(dataset)

            # Exit if video not opened.
            if not video.isOpened():
                print("Could not open video for dataset", dataset)
                sys.exit()

            self.videos[dataset] = video
            self.length = int(getVideo(dataset).get(cv2.CAP_PROP_FRAME_COUNT))

        for i in [0, 1]:
            # init frames
            self.updateFrames(0)
            self.updateFrames(1)

            # callback
            self.Visor[i].setCallback(lambda event, x, y, flags, dataset, li=i: self.clickEvent(li, event, x, y, flags, dataset))

        self.updateViews()
        # loop
        while True:

            k = self.Visor[0].getKey(0)

            frame_index = self.indexes[self.lastdataset]
            if k == 27:
                break
            elif k == 83 or k == 100:  # right || d
                frame_index += 1
            elif k == 81 or k == 97:  # left || a
                frame_index -= 1
            elif k == 82 or k == 119:  # up || w
                frame_index += 10
            elif k == 84 or k == 115:  # down || s
                frame_index -= 10
            elif k == 101:  # e
                frame_index += 100
            elif k == 113:  # q
                frame_index -= 100
            elif k == 80:  # start
                frame_index = 0
            elif k == 87:  # end
                frame_index = self.length - 1
            elif 49 <= k <= 57:  # 1-9
                frame_index = int(self.length * (k - 48) / 10.)
            elif k == 194: # F5
                reloadModule()
                self.updateViews()

            frame_index = max(0, min(self.length - 1, frame_index))
            if frame_index != self.indexes[self.lastdataset]:
                self.indexes[self.lastdataset] = frame_index
                self.updateFrames(self.lastdataset)

                self.updateViews()

    def clickEvent(self, ab, event, x, y, flags, dataset):
        # print event, flags
        # print ab

        self.lastdataset = ab
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
            self.Visor[i].setFrames(self.frames[i], copy=True)
            self.data[i].draw(self.Visor[i], C_RED)

        self.computeSimilarity()

        for i in (0, 1):
            self.Visor[i].showAll()

    def updateFrames(self, i):
        for dataset in self.groupDataset:
            self.videos[dataset].set(cv2.CAP_PROP_POS_FRAMES, self.indexes[i])
            ok, frame = self.videos[dataset].read()
            if not ok:
                print("Cannot read video file, dataset=", dataset, ",frame=", self.frames[i])
                sys.exit()
            self.frames[i][dataset] = frame

    def computeSimilarity(self):
        # get bboxes
        bboxA, datasetA = self.data[0].getBbox()
        bboxB, datasetB = self.data[1].getBbox()

        if bboxA is None or bboxB is None:
            return

        # get patches
        imageA = cutImage(self.frames[0][datasetA], bboxA)
        imageB = cutImage(self.frames[1][datasetB], bboxB)

        if imageA is None or imageB is None:
            return

        # get masks
        maskA = createMaskFromImage(imageA)
        maskB = createMaskFromImage(imageB)

        # compute similarity
        score, data = fdl.IstheSamePerson(imageA, maskA, imageB, maskB, False, fdl)

        # draw score
        valid = score == 7
        self.Visor[0].drawText(score, self.Visor[0].FLOOR, Point2D(0, 512), size=10, color=C_GREEN if valid else C_RED)
        if valid:
            for i in (0, 1): self.data[i].draw(self.Visor[i], C_GREEN)
            # cv2.imshow("test1", cv2.bitwise_or(imageA, imageA, mask=maskA))
            # cv2.imshow("test2", cv2.bitwise_or(imageB, imageB, mask=maskB))

        # draw data
        pos = 510. / len(data)
        thresholds = {'pA_ratio': 0.3, 'pB_ratio': 0.3, 'pB_shape': 1.5, 'pA_shape': 1.5, 'dist_r': -0.05, 'dist_g': -0.05, 'dist_b': -0.05}
        for k in sorted(data):
            text = k + "=" + str(data[k])
            valid = data[k] > thresholds[k] if thresholds[k] > 0 else data[k] < -thresholds[k]
            self.Visor[1].drawText(text, self.Visor[1].FLOOR, Point2D(0, pos), size=1, color=C_GREEN if valid else C_RED)
            pos += 510. / len(data)

        # draw masks
        self.Visor[0].drawImage(cv2.bitwise_or(imageA, imageA, mask=maskA), bboxA, datasetA)
        self.Visor[1].drawImage(cv2.bitwise_or(imageB, imageB, mask=maskB), bboxB, datasetB)


if __name__ == '__main__':
    CalibrationParser(getGroupedDatasets()['Laboratory/6p'])

    # list = getGroupedDatasets().values()
    # list.reverse()
    # for dataset in list:
    #     print(dataset)
    #     CalibrationParser(dataset)
