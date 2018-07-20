"""
Implementation of the algorithm. Main file
"""
import sys

import numpy as np
# from cachedDetectron import CachedDetectron as Detectron
from detectron import Detectron

# import cv2
import epfl_scripts.Utilities.cv2Visor as cv2
from epfl_scripts.Utilities.cache import cache_function
from epfl_scripts.Utilities.colorUtility import getColors
from epfl_scripts.Utilities.cv2Trackers import getTracker
from epfl_scripts.Utilities.geometry_utils import f_iou, f_euclidian, f_multiply, Point2D, Bbox, f_average
from epfl_scripts.groundTruthParser import getVideo, getGroupedDatasets, getCalibrationMatrix

WIN_NAME = "Tracking"

cv2.configure(100)

IOU_THRESHOLD = 0.5  # minimum iou to assign a detection to a prediction
FRAMES_LOST = 25  # maximum number of frames until the detection is removed

FRAMES_PERSON = 50  # after this number of frames with at least one camera tracking a target (id<0), it is assigned as person (id>=0)

CLOSEST_DIST = 20  # if an existing point is closer than this to a different point id, it is assigned that same id
FARTHEST_DIST = 50  # if an existing point is farther than the rest of the group, it is removed
DETECTION_DIST = 50  # if a new point is closer than this to an existing point, it is assigned the same id

FRAMES_CHANGEID = 5  # if this number of frames passed wanting to change id, it is changed


def getUnusedId(group):
    free = -1
    while free in group:
        free -= 1
    return free


class Prediction:
    def __init__(self, tracker=None, bbox=None, framesLost=0, trackerFound=False):
        self.tracker = tracker
        self.bbox = bbox
        self.framesLost = framesLost
        self.trackerFound = trackerFound

        self.newid = None
        self.newidCounter = 0

    def reset(self):
        self.__init__()


def findCenterOfGroup(predictions, id, cameras):
    points = []
    weights = []
    for camera in cameras:
        if predictions[camera][id].bbox is None: continue
        points.append(to3dWorld(camera, predictions[camera][id].bbox))
        weights.append(-predictions[camera][id].framesLost)
    return f_average(points, weights), len(points)


def estimateFromPredictions(predictions, ids, maxid, detector, cameras, frames):
    # predictions[camera][id] = prediction class
    # for id in ids: etc
    # detector[camera] = [ (bbox), ... ]
    # for camera in cameras: etc

    detector_unused = {}

    # phase 1: update bbox with detection (if available)
    if detector is not None:
        # assign detector instances to predictions
        for camera in cameras:
            detector_unused[camera] = detector[camera][:]
            for id in [x for x in ids if x >= 0] + [x for x in ids if x < 0]:
                prediction = predictions[camera][id]
                if prediction.bbox is None: continue

                # find closest detector to tracker
                bestBbox = prediction.bbox
                bestIoU = IOU_THRESHOLD
                for detBbox in detector[camera]:
                    iou = f_iou(prediction.bbox, detBbox)
                    if iou > bestIoU:
                        bestIoU = iou
                        bestBbox = detBbox

                if bestIoU > IOU_THRESHOLD:
                    # prediction found, remove from detections but keep for others
                    if bestBbox in detector_unused[camera]:
                        detector_unused[camera].remove(bestBbox)
                    prediction.trackerFound = True
                else:
                    # not found, lost if detector was active
                    prediction.trackerFound = False

                # update prediction
                if bestBbox != prediction.bbox:
                    # if not 'advanced' tracker, remove
                    if not hasattr(prediction.tracker, 'redefine'):
                        prediction.tracker = None
                prediction.bbox = bestBbox

    # phase 2: remove if lost enough times
    for camera in cameras:
        for id in ids:
            prediction = predictions[camera][id]

            if prediction.bbox is not None and prediction.trackerFound:
                framesLost = min(0, prediction.framesLost - 1)
            else:
                framesLost = max(0, prediction.framesLost + 1)

            if framesLost < FRAMES_LOST * (2 if id >= 0 else 1):
                prediction.framesLost = framesLost
            else:
                predictions[camera][id] = Prediction()

    # phase 3: assign unused detections to new targets
    if detector is not None:
        for camera in cameras:
            for bbox in detector_unused[camera]:
                # check if point is inside other detections in all cameras
                point3d = to3dWorld(camera, bbox)
                supported = True
                for camera2 in cameras:
                    if camera2 == camera: continue

                    valid = False
                    point2d = from3dWorld(camera2, point3d)
                    if not isInsideFrame(point2d, frames[camera]): continue
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
                    predictions[camera2][newid] = Prediction()
                predictions[camera][newid] = Prediction(bbox=bbox)
                if newid not in ids:
                    ids.append(newid)

    # phase 4: update ids individually
    centers = {}
    for id in ids:
        if id < 0: continue
        centers[id] = findCenterOfGroup(predictions, id, cameras)

    for i, id in enumerate(ids[:]):
        for camera in cameras:
            prediction = predictions[camera][id]
            if prediction.bbox is None: continue
            point3d = to3dWorld(camera, prediction.bbox)

            if id >= 0:
                # phase 4.1: find distance to group center, if far enough, remove
                center, elements = centers[id]
                if center is not None:
                    dist = f_euclidian(center, point3d)
                    if dist > FARTHEST_DIST:
                        if prediction.newidCounter > FRAMES_CHANGEID and prediction.newid is not None:
                            newid = getUnusedId(ids)
                            predictions[camera][newid] = prediction
                            predictions[camera][id] = Prediction()
                            prediction.newid = None
                            prediction.newidCounter = 0
                            prediction.framesLost = 0
                            for camera2 in cameras:
                                if newid in predictions[camera2]: continue
                                predictions[camera2][newid] = Prediction()
                            if newid not in ids:
                                ids.append(newid)
                            id = newid
                        else:
                            if prediction.newid is not None:
                                prediction.newidCounter += elements * 1. / len(cameras)
                            else:
                                prediction.newid = 'any'
                                prediction.newidCounter = 0
                    else:
                        if prediction.newid is not None:
                            prediction.newidCounter = 0
                            prediction.newid = None

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
                if prediction.newidCounter > FRAMES_CHANGEID and prediction.newid == bestId:
                    predictions[camera][bestId] = predictions[camera][id]
                    predictions[camera][id] = Prediction()
                    prediction.newid = None
                    prediction.newidCounter = 0
                    id = bestId
                else:
                    if prediction.newid == bestId:
                        prediction.newidCounter += bestElements * 1. / len(cameras)
                    else:
                        prediction.newid = bestId
                        prediction.newidCounter = 0
            else:
                if prediction.newid >= 0:
                    prediction.newidCounter = 0
                    prediction.newid = None

    # phase 5: set target as person if tracked continuously
    for id in ids[:]:
        if id >= 0: continue
        for camera in cameras:
            if predictions[camera][id].framesLost < -FRAMES_PERSON:
                if predictions[camera][id].newid >= 0:
                    newid = predictions[camera][id].newid
                else:
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

    # TODO: average point, move rectangles

    return predictions, ids, maxid


def to3dWorld(camera, bbox):
    calib = getCalibrationMatrix(camera)
    point = Point2D(bbox.xmin + bbox.width / 2., bbox.ymax)
    return f_multiply(calib, point)


def from3dWorld(camera, point):
    invCalib = np.linalg.inv(getCalibrationMatrix(camera))
    return f_multiply(invCalib, point)


def isInsideFrame(point, frame):
    x, y = point.getAsXY()
    height, width, colors = frame.shape

    return 0 <= x < width and 0 <= y < height


def fixbbox(frame, bbox):
    if bbox is None: return None

    height, width, colors = frame.shape

    # (0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows)
    if bbox.width <= 0 \
            or bbox.height <= 0:
        # bad bbox
        return None

    if bbox.xmin < 0:
        bbox.changeXmin(0)
    if bbox.ymin < 0:
        bbox.changeYmin(0)
    if bbox.xmax > width:
        bbox.changeXmax(width)
    if bbox.ymax > height:
        bbox.changeYmax(height)

    return bbox if bbox.isValid() else None


@cache_function("evalMultiTracker_{0}_{1}_{DETECTOR_FIRED}", lambda _gd, _tt, display, DETECTOR_FIRED: cache_function.TYPE_DISABLE if display else cache_function.TYPE_NORMAL, 8)
def evalMultiTracker(groupDataset, tracker_type, display=True, DETECTOR_FIRED=5):
    detector = {}  # detector[dataset][frame][index]

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
            print "Could not open video for dataset", dataset
            sys.exit()

        nframes = max(nframes, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))

        # Read first frame.
        ok, frame = video.read()
        frames[dataset] = frame
        if not ok:
            print "Cannot read video file"
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
                    except Exception:
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
                bbox = fixbbox(frames[dataset], estimations[dataset][id].bbox)
                estimations[dataset][id].bbox = bbox

                tracker = estimations[dataset][id].tracker

                # new bbox without tracker, initialize one
                if bbox is not None and tracker is None:
                    # intialize tracker
                    tracker = getTracker(tracker_type)
                    try:
                        tracker.init(frames[dataset], bbox.getAsXmYmWH())
                        estimations[dataset][id].tracker = tracker
                    except BaseException:
                        print "Error on tracker init"

                # if 'advanced' tracker with bbox, update
                if bbox is not None and tracker is not None and hasattr(tracker, 'redefine'):
                    tracker.redefine(bbox.getAsXmYmWH())

        # Show bounding boxes
        for dataset in groupDataset:
            data_detected[dataset][frame_index] = {}
            for id in ids:
                bbox = estimations[dataset][id].bbox
                if bbox is None: continue

                if id >= 0:
                    data_detected[dataset][frame_index][id] = bbox.getAsXmYmXMYM()  # xmin, ymin, xmax, ymax

                # show bbox
                if display:
                    label = "{0}:{1}:{2}".format(id, estimations[dataset][id].framesLost, estimations[dataset][id].newid)
                    p1 = (int(bbox.xmin), int(bbox.ymin))
                    p2 = (int(bbox.xmax), int(bbox.ymax))
                    color = colors[id % len(colors)] if id >= 0 else (255, 255, 255)
                    cv2.rectangle(frames[dataset], p1, p2, color, 2 if id >= 0 else 1, 1)
                    cv2.putText(frames[dataset], label, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.4 if id >= 0 else 0.2, color, 1)

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
            for dataset in groupDataset:
                for id in ids:
                    bbox = estimations[dataset][id].bbox
                    if bbox is None: continue

                    px, py = to3dWorld(dataset, bbox).getAsXY()
                    color = colors[id % len(colors)] if id >= 0 else (255, 255, 255)
                    thick = 2 if id >= 0 else 1
                    cv2.circle(frame, (int(px), int(py)), 1, color, thick)
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
    print ""
    if display:
        cv2.destroyWindow(WIN_NAME)

    for dataset in groupDataset:
        videos[dataset].release()

    return frame_index, range(maxid + 1), data_detected


if __name__ == '__main__':
    # dataset = getGroupedDatasets()['Terrace/terrace1']
    # dataset = getGroupedDatasets()['Passageway/passageway1']
    dataset = getGroupedDatasets()['Laboratory/6p']
    # dataset = getGroupedDatasets()['Campus/campus7']

    # tracker = 'BOOSTING'  # slow good
    tracker = 'KCF'  # fast bad
    # tracker = 'MYTRACKER'  # with redefine

    detector_fired = 5

    print dataset, tracker, detector_fired
    evalMultiTracker(dataset, tracker, True, detector_fired)
