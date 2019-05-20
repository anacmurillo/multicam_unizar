from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
from __future__ import print_function

from epfl_scripts.Debugging.calibrationParser import toInt
from epfl_scripts.Utilities.cilinderTracker import CilinderTracker
from epfl_scripts.Utilities.cilinderTracker import to3dCilinder
from epfl_scripts.Utilities.geometry3D_utils import Cilinder, f_averageCilinders

"""
Implementation of the algorithm. Main file
"""

try:
    from detectron import Detectron
except Exception as e:
    print("Detectron not available, using cached one. Full exception below:")
    print(e)
    from cachedDetectron import CachedDetectron as Detectron

import sys

import numpy as np

# import cv2
import epfl_scripts.Utilities.cv2Visor as cv2

cv2.configure(100)

from epfl_scripts.Utilities.cache import cache_function
from epfl_scripts.Utilities.colorUtility import getColors
from epfl_scripts.Utilities.cv2Trackers import getTracker
from epfl_scripts.Utilities.geometry2D_utils import f_iou, f_euclidian, f_multiply, Point2D, Bbox, f_average, f_subtract, f_area, f_multiplyInv, f_add
from epfl_scripts.groundTruthParser import getVideo, getGroupedDatasets, getCalibrationMatrix, getCalibrationMatrixFull

WIN_NAME = "Tracking"
WINF_NAME = WIN_NAME + "_overview"

DIST_THRESHOLD = 20  # minimum dist to assign a detection to a prediction (in floor plane coordinates)
HEIGHT_THRESHOLD = 0.5
WIDTH_THRESHOLD = 0.5

FRAMES_LOST = 25  # maximum number of frames until the detection is removed

FRAMES_PERSON = 50  # after this number of frames with at least one camera tracking a target (id<0), it is assigned as person (id>=0)

CLOSEST_DIST = 20  # if an existing point is closer than this to a different point id, it is assigned that same id (in floor plane coordinates)
FARTHEST_DIST = 50  # if an existing point is farther than the rest of the group, it is removed (in floor plane coordinates)
DETECTION_DIST = 50  # if a new point is closer than this to an existing point, it is assigned the same id (in floor plane coordinates)

FRAMES_CHANGEID = 5  # if this number of frames passed wanting to change id, it is changed


def getUnusedId(group):
    free = -1
    while free in group:
        free -= 1
    return free


class Prediction:
    NEXTUID = 0

    def __init__(self, tracker=None):
        self.tracker = tracker
        self.framesLost = 0
        self.trackerFound = False
        self.detectorFound = False

        self.person = False
        self.uniqueID = -1

        self.newid = None
        self.newidCounter = 0
        self.newTags = set()

        self.redefineCilinder = None

    def redefine(self, cilinder):
        self.redefineCilinder = cilinder

    def updateCilinder(self, images):
        # cilinders = []
        # weights = []
        #
        # cilinders.append(self.tracker.getCilinder())
        # weights.append(2)
        #
        # for cilinder in self.cilinders:
        #     cilinders.append(cilinder)
        #     weights.append(1)
        #
        # ok, new_point = self.tracker.update(images, f_averageCilinders(cilinders, weights))
        #
        # self.trackerFound = ok
        # self.cilinders.clear()
        ##################

        self.trackerFound, _ = self.tracker.update(images, self.redefineCilinder)

    def setPersonIfRequired(self):
        if self.person: return

        if self.framesLost < -FRAMES_PERSON:
            self.person = True
            self.uniqueID = Prediction.NEXTUID
            Prediction.NEXTUID += 1

    def updateLost(self, weight):
        if self.trackerFound and self.detectorFound:
            framesLost = min(0, self.framesLost - 1)  # if negative, found
        else:
            framesLost = max(0, self.framesLost + 1)  # if positive, lost

        if framesLost < FRAMES_LOST * weight:
            self.framesLost = framesLost
            return False
        else:
            # lost
            return True

    def wantToChange(self, tag, newId, counter):
        self.newTags.add(tag)

        same, newId = compareIds(self.newid, newId)
        if same:
            # we want to change again
            if self.newidCounter > FRAMES_CHANGEID:
                # a lot of frames since we want to change, lets change
                self.newTags.clear()
                self.newid = None
                self.newidCounter = 0
                self.framesLost = 0
                return newId
            else:
                # we want to change, but still not time enough
                self.newidCounter += counter
                self.newid = newId
                return None
        else:
            # first time we want to change, init
            self.newid = newId
            self.newidCounter = 0

    def dontWantToChange(self, tag):
        if tag in self.newTags:
            # wanted to change, don't want now
            self.newTags.remove(tag)
            if len(self.newTags) == 0:
                # no other thing want to change, stop
                self.newidCounter = 0
                self.newid = None

    def isWorstThan(self, other):
        """
        more lost is worst
        """

        # we are similar, and we have lost it more -> we are worst
        return f_similarCilinders(self.tracker.getCilinder(), other.tracker.getCilinder()) and self.framesLost > other.framesLost

def compareIds(id1, id2):
    """
    returns same, newid
        if same==True, id1 and id2 can be considered the same and newid is that common id
        else id1 and id2 are different, and newid=id2
    """
    if id1 is None and id2 is not None:
        return False, id2
    if id1 is not None and id2 is None:
        return False, id2
    if id1 is None and id2 is None:
        return True, None
    if id1 is 'any':
        return True, id2
    if id2 is 'any':
        return True, id1
    if id1 == id2:
        return True, id1
    return False, id2


def findCenterOfGroup(predictions, id, cameras):
    points = []
    weights = []
    for camera in cameras:

        # don't use if not a bbox
        bbox = predictions[camera][id].bbox
        if bbox is None: continue

        points.append(to3dPoint(camera, predictions[camera][id].bbox))
        weights.append(-predictions[camera][id].framesLost)
    return f_average(points, weights), len(points)


def f_similarCilinders(a, b):
    return f_euclidian(a.getCenter(), b.getCenter()) < DIST_THRESHOLD \
           and abs(a.getWidth() - b.getWidth()) < WIDTH_THRESHOLD \
           and abs(a.getHeight() - b.getHeight()) < HEIGHT_THRESHOLD


def estimateFromPredictions(predictions, detector, cameras, frames):
    # predictions = [prediction class]
    # detector[camera] = [ (bbox), ... ]
    # for camera in cameras: etc

    detector_unused = {}

    # phase 1: update bbox with detection (if available)
    if detector is not None:
        # assign detector instances to predictions

        for camera in cameras:
            detector_unused[camera] = detector[camera][:]

        for prediction in predictions:
            bboxes = prediction.tracker.getBboxes()

            allCilinders = []
            allWeights = []

            for camera in cameras:
                # update all predictions with the detections (if available)
                for detection in detector[camera]:

                    simil = f_iou(bboxes[camera], detection)  # getSimilarity(bboxes[camera], detection)

                    if simil > 0:
                        # use the detection
                        allCilinders.append(to3dCilinder(camera, detection))
                        allWeights.append(simil)

                        if detection in detector_unused[camera]: detector_unused[camera].remove(detection)

            if len(allCilinders) > 0:

                cilinder = f_averageCilinders(allCilinders, allWeights)

                prediction.redefine(cilinder)

                prediction.detectorFound = True
            else:
                prediction.detectorFound = False

    for prediction in predictions[:]:
        # phase 2.1: remove if lost enough times
        if prediction.updateLost(1 if prediction.person else 2):
            predictions.remove(prediction)

    # phase 3: assign unused detections to new targets
    if detector is not None:
        # for each bbox
        for camera in cameras:
            for bbox in detector_unused[camera]:
                group = {camera: bbox}
                cilinder = to3dCilinder(camera, bbox)
                # find other similar in other cameras
                for camera2 in cameras:
                    if camera == camera2: continue
                    for bbox2 in detector_unused[camera2][:]:
                        if f_similarCilinders(cilinder, to3dCilinder(camera2, bbox2)):
                            group[camera2] = bbox2
                            detector_unused[camera2].remove(bbox2)

                # create new tracker
                tracker = CilinderTracker(cameras)
                tracker.init(frames, group)
                if len(predictions) < 5 or True:
                    predictions.append(Prediction(tracker))

    # phase 4: remove worst trackers
    uniquePredictions = []
    for prediction in predictions:
        keep = True
        for prediction2 in predictions:
            if prediction.isWorstThan(prediction2):
                keep = False
        if keep:
            uniquePredictions.append(prediction)
    predictions = uniquePredictions


    # phase 5: set target as person if tracked continuously
    for prediction in predictions:
        prediction.setPersonIfRequired()

    return predictions


def to3dPoint(camera, bbox):
    calib = getCalibrationMatrix(camera)
    return f_multiply(calib, bbox.getFeet())


def from3dPoint(camera, point):
    calib = getCalibrationMatrix(camera)
    return f_multiplyInv(calib, point)


def isInsideFrameP(point, frame):
    x, y = point.getAsXY()
    height, width, _colors = frame.shape

    return 0 <= x < width and 0 <= y < height


def isInsideFrameB(bbox, frame):
    height, width, _colors = frame.shape
    return bbox.xmin > 0 and bbox.xmax < width and bbox.ymax < height  # bbox.ymin > 0 not checked


def cropBbox(bbox, frame):
    if bbox is None: return None

    height, width, colors = frame.shape

    # (0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows)

    return Bbox.XmYmXMYM(max(round(bbox.xmin), 0), max(round(bbox.ymin), 0), min(round(bbox.xmax), width), min(round(bbox.ymax), height))


@cache_function("evalMultiTracker_{0}_{1}_{DETECTOR_FIRED}", lambda _gd, _tt, display, DETECTOR_FIRED: cache_function.TYPE_DISABLE if display else cache_function.TYPE_NORMAL, 8)
def evalMultiTracker(groupDataset, tracker_type, display=True, DETECTOR_FIRED=5):
    # colors
    colors = getColors(12)

    # get detector
    detector = Detectron()

    # floor
    if display:
        frame_floor = np.zeros((512, 512, 3), np.uint8)

    # initialize videos
    videos = {}
    frames = {}
    nframes = 0
    for dataset in groupDataset:
        video = getVideo(dataset)
        videos[dataset] = video

        # Exit if video not opened.
        if not video.isOpened():
            print("Could not open video for dataset", dataset)
            sys.exit()

        nframes = max(nframes, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))

        # Read first frame.
        ok, frame = video.read()
        frames[dataset] = frame
        if not ok:
            print("Cannot read video file for dataset", dataset)
            sys.exit()

        if display:
            # draw floor visibility lines
            height, width, _ = frame.shape

            matrix = getCalibrationMatrix(dataset)

            tl = toInt(f_multiply(matrix, Point2D(0, height / 2)).getAsXY())
            tr = toInt(f_multiply(matrix, Point2D(width, height / 2)).getAsXY())
            bl = toInt(f_multiply(matrix, Point2D(0, height)).getAsXY())
            br = toInt(f_multiply(matrix, Point2D(width, height)).getAsXY())

            color = (100, 100, 100)

            # cv2.line(self.frames[FLOOR], tl, tr, color, 1, 1)
            cv2.line(frame_floor, tr, br, color, 1, 1)
            cv2.line(frame_floor, bl, br, color, 1, 1)
            cv2.line(frame_floor, tl, bl, color, 1, 1)

    # initialize detection set
    data_detected = {}
    for dataset in groupDataset:
        data_detected[dataset] = {}

    # initialize predictions
    predictions = []

    # initialize windows
    cv2.namedWindow(WIN_NAME)
    cv2.moveWindow(WIN_NAME, 0, 0)
    cv2.namedWindow(WINF_NAME)
    cv2.moveWindow(WINF_NAME, 0, frame.shape[1]+100)

    # loop
    frame_index = 0
    allOk = True
    while allOk:

        # parse trackers
        for prediction in predictions:
            prediction.updateCilinder(frames)

        # run detector
        if frame_index % DETECTOR_FIRED == 0:
            detector_results = {}
            for dataset in groupDataset:
                results = detector.evaluateImage(frames[dataset], str(dataset) + " - " + str(frame_index))
                detector_results[dataset] = [Bbox.XmYmXMYM(result[0], result[1], result[2], result[3]) for result in results]

            # show detections
            if display:
                color = (100, 100, 100)
                for dataset in groupDataset:
                    for bbox in detector_results[dataset]:
                        tl = (int(bbox.xmin), int(bbox.ymin))
                        br = (int(bbox.xmax), int(bbox.ymax))
                        cl = (int(bbox.xmin), int(bbox.ymin + bbox.height / 2))
                        cv2.rectangle(frames[dataset], tl, br, color, 1, 1)
        else:
            detector_results = None

        # merge all predictions -> estimations
        estimations = estimateFromPredictions(predictions, detector_results, groupDataset, frames)

        # compute detections
        for estimation in estimations:
            bboxes = estimation.tracker.getBboxes()
            for dataset in groupDataset:
                data_detected[dataset][frame_index] = {}

                bbox = cropBbox(bboxes[dataset], frames[dataset])
                if bbox is None: continue

                if estimation.person:
                    data_detected[dataset][frame_index][estimation.uniqueID] = bbox.getAsXmYmXMYM()  # xmin, ymin, xmax, ymax

                # show bbox
                if display:
                    # label = "{0}:{1}:{2}".format(id, estimations[dataset][id].framesLost, estimations[dataset][id].newid)
                    label = "{0}:{1}:{2}".format(estimation.uniqueID, estimation.framesLost, estimation.newid)
                    tl = (int(bbox.xmin), int(bbox.ymin))
                    br = (int(bbox.xmax), int(bbox.ymax))
                    cl = (int(bbox.xmin), int(bbox.ymin + bbox.height / 2))
                    color = colors[estimation.uniqueID % len(colors)] if estimation.person else (255, 255, 255)
                    cv2.rectangle(frames[dataset], tl, br, color, 2 if estimation.person else 1, 1)
                    cv2.putText(frames[dataset], label, cl, cv2.FONT_HERSHEY_SIMPLEX, 0.4 if estimation.person else 0.35, color, 1)

        if display:
            # Display result
            concatenatedFrames = None
            for dataset in groupDataset:
                if concatenatedFrames is None:
                    concatenatedFrames = frames[dataset]
                else:
                    concatenatedFrames = np.hstack((concatenatedFrames, frames[dataset]))
            cv2.imshow(WIN_NAME, concatenatedFrames)

            # display overview
            frame = frame_floor.copy()
            for estimation in estimations:
                color = colors[estimation.uniqueID % len(colors)] if estimation.person else (255, 255, 255)
                thick = 2 if estimation.person else 1

                # each center
                center = estimation.tracker.getCilinder().getCenter()
                x, y = center.getAsXY()
                cv2.drawMarker(frame, (int(x), int(y)), (0, 0, 0), 3, 4)
                cv2.drawMarker(frame, (int(x), int(y)), color, 0, 2)

                # with radius
                radius = estimation.tracker.getCilinder().getWidth()
                cv2.circle(frame, (int(x), int(y)), int(radius), color, thickness=1, lineType=8)

            cv2.putText(frame, str(frame_index), (0, 512), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(WINF_NAME, frame)
            # /end display overview

            if cv2.waitKey(1) & 0xff == 27:
                break
        else:
            # show progress
            # if sys.stdout.isatty():
            sys.stdout.write("\r" + str(frame_index) + "/" + str(nframes) + "     ")
            sys.stdout.flush()

        # read new frames
        for dataset in groupDataset:
            # Read a new frame
            ok, frames[dataset] = videos[dataset].read()
            allOk = allOk and ok

        predictions = estimations

        frame_index += 1

    # clean
    print("")
    if display:
        cv2.destroyWindow(WIN_NAME)
        cv2.destroyWindow(WINF_NAME)

    for dataset in groupDataset:
        videos[dataset].release()

    return frame_index, range(Prediction.NEXTUID), data_detected


if __name__ == '__main__':
    # choose dataset
    # dataset = getGroupedDatasets()['Terrace/terrace1']
    # dataset = getGroupedDatasets()['Passageway/passageway1']
    dataset = getGroupedDatasets()['Laboratory/6p']
    # dataset = getGroupedDatasets()['Campus/campus7']

    # v = dataset[0]
    # dataset = [v]

    # choose tracker
    # tracker = 'BOOSTING'  # slow good
    # tracker = 'KCF'  # fast bad
    # tracker = 'MYTRACKER'  # with redefine
    tracker = 'DEEP_SORT'  # deep sort

    # choose parameter
    detector_fired = 1

    # run
    print(dataset, tracker, detector_fired)
    evalMultiTracker(dataset, tracker, True, detector_fired)
