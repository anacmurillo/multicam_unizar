from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
from __future__ import print_function

from epfl_scripts.Utilities.geometry3D_utils import Cilinder

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

IOU_THRESHOLD = 0.5  # minimum iou to assign a detection to a prediction
FRAMES_LOST = 25  # maximum number of frames until the detection is removed

FRAMES_PERSON = 50  # after this number of frames with at least one camera tracking a target (id<0), it is assigned as person (id>=0)

CLOSEST_DIST = 20  # if an existing point is closer than this to a different point id, it is assigned that same id
FARTHEST_DIST = 50  # if an existing point is farther than the rest of the group, it is removed
DETECTION_DIST = 50  # if a new point is closer than this to an existing point, it is assigned the same id

FRAMES_CHANGEID = 5  # if this number of frames passed wanting to change id, it is changed

REDEFINED_PARAM = 0  # percent of the redefined bbox (1=new, 0=original)
CENTERBBOX_PARAM = 0  # percent of the centering of the bboxes (1=center, 0=unchanged)


def getUnusedId(group):
    free = -1
    while free in group:
        free -= 1
    return free


class Prediction:
    def __init__(self, tracker=None, bbox=None, framesLost=0, trackerFound=False, detectorFound=False):
        self.tracker = tracker
        self.bbox = bbox
        self.framesLost = framesLost
        self.trackerFound = trackerFound
        self.detectorFound = detectorFound

        self.newid = None
        self.newidCounter = 0
        self.newTags = set()

    def reset(self):
        self.__init__()

    def setBbox(self, newBbox):
        if newBbox != self.bbox:
            if hasattr(self.tracker, 'redefine'):
                self.tracker.redefine(newBbox.getAsXmYmWH(), REDEFINED_PARAM)
            else:
                # if not 'advanced' tracker, remove
                self.tracker = None
            self.bbox = newBbox

    def updateLost(self, weight):
        if self.bbox is not None and self.trackerFound and self.detectorFound:
            framesLost = min(0, self.framesLost - 1)  # if negative, found
        else:
            framesLost = max(0, self.framesLost + 1)  # if positive, lost

        if framesLost < FRAMES_LOST * weight:
            self.framesLost = framesLost
        else:
            self.reset()

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


def estimateFromPredictions(predictions, ids, maxid, detector, cameras, frames):
    # predictions[camera][id] = prediction class
    # for id in ids: etc
    # detector[camera] = [ (bbox), ... ]
    # for camera in cameras: etc

    # phase 0: redefine bbox with centers if advanced trackers
    if CENTERBBOX_PARAM > 0:
        for id in ids:
            if id < 0: continue
            # calculate center of groups
            center, _ = findCenterOfGroup(predictions, id, cameras)
            if center is None: continue

            for camera in cameras:
                bbox = predictions[camera][id].bbox
                if bbox is not None:
                    # there is a bbox, redefine
                    p_center = from3dPoint(camera, center)
                    p_bbox = bbox.getFeet()
                    bbox.translate(f_subtract(p_center, p_bbox).getAsXY(), CENTERBBOX_PARAM)
                    if not hasattr(predictions[camera][id].tracker, 'redefine'):
                        predictions[camera][id].tracker = None
                # else:
                # there is no Detection, create new one
                # error, can't create because we don't know the height/width!
                # predictions[camera][id] = Prediction()

    detector_unused = {}

    # phase 1: update bbox with detection (if available)
    if detector is not None:
        # assign detector instances to predictions
        for camera in cameras:
            detector_unused[camera] = detector[camera][:]
            for id in [x for x in ids if x >= 0] + [x for x in ids if x < 0]:
                prediction = predictions[camera][id]
                if prediction.bbox is None: continue
                bbox = cropBbox(prediction.bbox, frames[camera])

                # find closest detector to tracker
                bestBbox = None
                bestIoU = IOU_THRESHOLD
                for detBbox in detector[camera]:
                    iou = f_iou(bbox, detBbox)
                    if iou > bestIoU:
                        bestIoU = iou
                        bestBbox = detBbox

                if bestIoU > IOU_THRESHOLD:
                    # prediction found, remove from detections but keep for others
                    if bestBbox in detector_unused[camera]:
                        detector_unused[camera].remove(bestBbox)
                    prediction.detectorFound = True

                    # update prediction
                    prediction.setBbox(bestBbox)

                else:
                    # not found, lost if detector was active
                    prediction.detectorFound = False

    for camera in cameras:
        for id in ids[:]:
            # phase 2.1: remove if lost enough times
            predictions[camera][id].updateLost(2 if id >= 0 else 1)

            bbox = predictions[camera][id].bbox
            if bbox is None: continue

            # phase 2.2: remove if empty area
            if f_area(bbox) <= 0:
                predictions[camera][id].reset()
                continue

            # phase 2.3: remove if same as other prediction
            if id < 0:
                for id2 in ids:
                    if id2 <= id: continue
                    # check for other ids (including persons)

                    bbox2 = predictions[camera][id2].bbox
                    if bbox2 is None: continue
                    if f_iou(bbox, bbox2) > IOU_THRESHOLD:
                        # same as other prediction, remove
                        predictions[camera][id].reset()

    # phase 3: assign unused detections to new targets
    if detector is not None:
        for camera in cameras:
            for bbox in detector_unused[camera]:
                # check if point is inside other detections in all cameras
                point3d = to3dPoint(camera, bbox)
                supported = True
                for camera2 in cameras:
                    if camera2 == camera: continue

                    valid = False
                    point2d = from3dPoint(camera2, point3d)
                    if not isInsideFrameP(point2d, frames[camera]): continue
                    for bbox2 in detector[camera2]:
                        if bbox2.contains(point2d, 10):
                            valid = True
                            break
                    if not valid:
                        supported = False
                        break

                if not supported: continue

                newid = getUnusedId(ids)
                for camera2 in cameras:
                    if newid in predictions[camera2]: continue
                    predictions[camera2][newid] = Prediction()  # empty to avoid keyerror
                predictions[camera][newid] = Prediction(bbox=bbox)
                if newid not in ids:
                    ids.append(newid)

    # phase 4: update ids individually
    centers = {}
    for id in ids:
        if id < 0: continue
        # calculate center of groups
        centers[id] = findCenterOfGroup(predictions, id, cameras)

    for i, id in enumerate(ids[:]):
        # for each id

        for camera in cameras:
            # in each camera

            # get prediction in 3d world (if available)
            prediction = predictions[camera][id]
            if prediction.bbox is None: continue
            point3d = to3dPoint(camera, prediction.bbox)

            if id >= 0:

                # phase 4.1: find distance to group center, if far enough, remove
                center, elements = centers[id]
                if center is not None:
                    # there is a group center
                    dist = f_euclidian(center, point3d)
                    if dist > FARTHEST_DIST:
                        # we are far from the center
                        newid = prediction.wantToChange('far', 'any', elements * 1. / len(cameras))
                        if newid != id and newid is not None:
                            # lets change
                            if newid == 'any':
                                newid = getUnusedId(ids)
                            predictions[camera][newid] = prediction
                            predictions[camera][id] = Prediction()
                            for camera2 in cameras:
                                if newid in predictions[camera2]: continue
                                predictions[camera2][newid] = Prediction()
                            if newid not in ids:
                                ids.append(newid)
                            id = newid
                    else:
                        # not far from the center
                        prediction.dontWantToChange('far')

            # phase 4.2: find closest other point, if other id change to that
            bestId = None
            bestElements = 0
            bestDist = CLOSEST_DIST
            for id2 in ids[:]:
                if id2 < 0: continue
                center, elements = centers[id2]
                if center is not None:
                    dist = f_euclidian(center, point3d)

                    if dist < bestDist:
                        bestId = id2
                        bestDist = dist
                        bestElements = elements
            if bestId is not None and bestId != id:
                # found a closer point
                newid = prediction.wantToChange('closer', bestId, bestElements * 1. / len(cameras))
                if newid != id and newid is not None:
                    # lets change
                    if newid == 'any':
                        newid = getUnusedId(ids)
                    predictions[camera][newid] = prediction
                    predictions[camera][id] = Prediction()
                    for camera2 in cameras:
                        if newid in predictions[camera2]: continue
                        predictions[camera2][newid] = Prediction()
                    if newid not in ids:
                        ids.append(newid)
                    id = newid
            else:
                # there isn't a closer point
                prediction.dontWantToChange('closer')

    # phase 5: set target as person if tracked continuously
    for id in ids[:]:
        if id >= 0: continue
        for camera in cameras:
            newid = predictions[camera][id].newid
            if predictions[camera][id].framesLost < -FRAMES_PERSON / (1. if newid is None else 2.):
                if newid is None or newid == 'any':
                    newid = maxid + 1
                    maxid = newid
                predictions[camera][newid] = predictions[camera][id]
                predictions[camera][id] = Prediction()
                for camera2 in cameras:
                    if newid in predictions[camera2]: continue
                    predictions[camera2][newid] = Prediction()
                if newid not in ids:
                    ids.append(newid)

    # internal phase 6: calculate dispersion of each id and remove empty ones
    newids = []
    for id in ids:
        #  maxdist = 0
        points = 0
        for i, camera in enumerate(cameras):
            if predictions[camera][id].bbox is None: continue
            points += 1
            # for camera2 in cameras[0:i]:
            #     if camera == camera2: continue
            #     if predictions[camera2][id].bbox is None: continue
            #
            #     dist = f_euclidian(to3dWorld(camera, predictions[camera][id].bbox), to3dWorld(camera2, predictions[camera2][id].bbox))
            #     if dist > maxdist:
            #         maxdist = dist

        #  if points > 1:
        #    print "id=", id, " maxdist=", maxdist, " points=", points

        if points > 0:
            newids.append(id)
    ids = newids

    return predictions, ids, maxid


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
            print("Cannot read video file")
            sys.exit()

    # initialize detection set
    data_detected = {}
    for dataset in groupDataset:
        data_detected[dataset] = {}

    # initialize predictions
    predictions = {}
    ids = []
    for dataset in groupDataset:
        predictions[dataset] = {}

    # loop
    frame_index = 0
    allOk = True
    maxid = -1
    while allOk:

        # parse trackers
        for dataset in groupDataset:
            for id in ids:
                tracker = predictions[dataset][id].tracker
                if tracker is not None:
                    # get tracker prediction
                    try:
                        ok, bbox = tracker.update(frames[dataset])
                    except Exception as ex:
                        print("Exception when updating tracker:", ex)
                        ok = False
                        bbox = None
                    if ok:
                        predictions[dataset][id].bbox = Bbox.XmYmWH(*bbox)
                        predictions[dataset][id].trackerFound = True
                    else:
                        predictions[dataset][id].trackerFound = False

        # detector needed
        if frame_index % DETECTOR_FIRED == 0:
            detector_results = {}
            for dataset in groupDataset:
                results = detector.evaluateImage(frames[dataset], str(dataset) + " - " + str(frame_index))
                detector_results[dataset] = [Bbox.XmYmXMYM(result[0], result[1], result[2], result[3]) for result in results]
        else:
            detector_results = None

        # merge all predictions -> estimations
        estimations, ids, maxid = estimateFromPredictions(predictions, ids, maxid, detector_results, groupDataset, frames)

        # initialize new trackers
        for dataset in groupDataset:
            for id in ids:
                bbox = estimations[dataset][id].bbox

                tracker = estimations[dataset][id].tracker

                # new bbox without tracker, initialize one
                if bbox is not None and tracker is None:
                    # intialize tracker
                    tracker = getTracker(tracker_type)
                    try:
                        tracker.init(frames[dataset], bbox.getAsXmYmWH())
                        estimations[dataset][id].tracker = tracker
                    except BaseException as e:
                        print("Error on tracker init", e)

        # Show bounding boxes
        for dataset in groupDataset:
            data_detected[dataset][frame_index] = {}
            for id in ids:
                bbox = cropBbox(estimations[dataset][id].bbox, frames[dataset])
                if bbox is None: continue

                if id >= 0:
                    data_detected[dataset][frame_index][id] = bbox.getAsXmYmXMYM()  # xmin, ymin, xmax, ymax

                # show bbox
                if display:
                    label = "{0}:{1}:{2}".format(id, estimations[dataset][id].framesLost, estimations[dataset][id].newid)
                    tl = (int(bbox.xmin), int(bbox.ymin))
                    br = (int(bbox.xmax), int(bbox.ymax))
                    cl = (int(bbox.xmin), int(bbox.ymin + bbox.height / 2))
                    color = colors[id % len(colors)] if id >= 0 else (255, 255, 255)
                    cv2.rectangle(frames[dataset], tl, br, color, 2 if id >= 0 else 1, 1)
                    cv2.putText(frames[dataset], label, cl, cv2.FONT_HERSHEY_SIMPLEX, 0.4 if id >= 0 else 0.35, color, 1)

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
            frame = np.zeros((512, 512, 3), np.uint8)
            for id in ids:
                color = colors[id % len(colors)] if id >= 0 else (255, 255, 255)
                thick = 2 if id >= 0 else 1

                # each point in camera
                for dataset in groupDataset:
                    bbox = estimations[dataset][id].bbox
                    if bbox is None: continue

                    px, py = to3dPoint(dataset, bbox).getAsXY()
                    cv2.circle(frame, (int(px), int(py)), 1, color, thick)

                # center
                if id >= 0:
                    center, _ = findCenterOfGroup(estimations, id, groupDataset)
                    if center is not None:
                        x, y = center.getAsXY()
                        cv2.drawMarker(frame, (int(x), int(y)), (0, 0, 0), 3, 4)
                        cv2.drawMarker(frame, (int(x), int(y)), color, 0, 2)

            cv2.putText(frame, str(frame_index), (0, 512), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(WIN_NAME + "_overview", frame)
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
        cv2.destroyWindow(WIN_NAME + "_overview")

    for dataset in groupDataset:
        videos[dataset].release()

    return frame_index, range(maxid + 1), data_detected


if __name__ == '__main__':
    # choose dataset
    # dataset = getGroupedDatasets()['Terrace/terrace1']
    # dataset = getGroupedDatasets()['Passageway/passageway1']
    dataset = getGroupedDatasets()['Laboratory/6p']
    # dataset = getGroupedDatasets()['Campus/campus7']

    # choose tracker
    # tracker = 'BOOSTING'  # slow good
    # tracker = 'KCF'  # fast bad
    # tracker = 'MYTRACKER'  # with redefine
    tracker = 'DEEP_SORT'  # deep sort

    # choose parameter
    detector_fired = 10

    # run
    print(dataset, tracker, detector_fired)
    evalMultiTracker(dataset, tracker, True, detector_fired)
