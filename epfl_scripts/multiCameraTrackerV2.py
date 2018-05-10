import numpy as np
import sys

# import cv2
import epfl_scripts.Utilities.cv2Visor as cv2
from epfl_scripts.Utilities.cache import cache_function
from epfl_scripts.Utilities.colorUtility import getColors
from epfl_scripts.Utilities.cv2Trackers import getTracker, getTrackers
from epfl_scripts.Utilities.geometry_utils import f_iou, f_euclidian, f_multiply, Point2D, Bbox
from epfl_scripts.Utilities.groundTruthParser import getVideo, getGroupedDatasets, getSuperDetector, getCalibrationMatrix

WIN_NAME = "Tracking"

cv2.configure(100)

DETECTOR_FIRED = 5  # after this number of frames the detector is evaluated

IOU_THRESHOLD = 0.5  # minimum iou to assign a detection to a prediction
FRAMES_LOST = 25  # maximum number of frames until the detection is removed

FRAMES_PERSON = 50  # after this number of frames with at least one camera tracking a target (id<0), it is assigned as person (id>=0)

CLOSEST_DIST = 20  # if an existing point is closer than this to a different point id, it is assigned that same id
FARTHEST_DIST = 50  # if an existing point is farther than the rest of the group, it is removed
DETECTION_DIST = 50  # if a new point is closer than this to an existing point, it is assigned the same id


def findfree(group, person):
    if person:
        return max(group + [-1]) + 1
    else:
        free = -1
        while free in group:
            free -= 1
        return free


def estimateFromPredictions(predictions, ids, detector, cameras):
    # predictions[camera][id] = [ tracker, bbox, framesLost, trackerInfo {add as necessary} ]
    # for id in ids: etc
    # detector[camera] = [ (bbox), ... ]
    # for camera in cameras: etc

    pos3d = []

    # phase 1: update bbox with detection (if available)
    if detector is not None:
        # assign detector instances to predictions
        for camera in cameras:
            detectorUsed = []
            for id in ids:
                tracker, bbox, framesLost, tracker_found = predictions[camera][id]
                if bbox is None: continue

                # find closest detector to tracker
                bestBbox = bbox
                bestIoU = IOU_THRESHOLD
                for detBbox in detector[camera] + detectorUsed:
                    iou = f_iou(bbox, detBbox)
                    if iou > bestIoU:
                        bestIoU = iou
                        bestBbox = detBbox

                if bestIoU > IOU_THRESHOLD:
                    # prediction found, remove from detections but keep for others
                    if bestBbox in detector[camera]:
                        detector[camera].remove(bestBbox)
                        # detectorUsed.append(bestBbox)
                    tracker_found = True
                else:
                    # not found, lost if detector was active
                    tracker_found = False

                # update bbox with best bbox (original or new)
                predictions[camera][id] = [tracker if bestBbox == bbox else None, bestBbox, framesLost, tracker_found]

    # phase 2: populate 3d info and remove if lost enough times
    for camera in cameras:
        for id in ids:
            tracker, bbox, framesLost, tracker_found = predictions[camera][id]

            if bbox is not None and tracker_found:
                framesLost = min(0, framesLost - 1)
            else:
                framesLost = max(0, framesLost + 1)

            if framesLost < FRAMES_LOST:
                predictions[camera][id][2] = framesLost

                # add 3d
                if bbox is not None:
                    pos3d.append((to3dWorld(camera, bbox), id))
            else:
                predictions[camera][id] = [None, None, 0, False]

    # phase 4: assign unused detections to new targets
    # TODO: initialize only if cameras support decision
    if detector is not None:
        for camera in cameras:
            for bbox in detector[camera]:

                id = findfree(ids, False)
                for dataset2 in cameras:
                    predictions[dataset2][id] = [None, None, 0, False]
                predictions[camera][id] = [None, bbox, 0, False]
                ids.append(id)

    # update ids individually
    # phase 2: change id to nearest
    for i, id in enumerate(ids[:]):
        for camera in cameras:
            tracker, bbox, framesLost, tracker_found = predictions[camera][id]
            if bbox is None: continue

            # phase 2.1: find closest id in group, if far enough, change
            # TODO: change to center of group
            # TODO change only if counter >= x
            bestDist = FARTHEST_DIST
            found = False
            points = 1
            for dataset2 in cameras:
                if dataset2 == camera: continue
                if predictions[dataset2][id][1] is None: continue
                points += 1
                dist = f_euclidian(to3dWorld(dataset2, predictions[dataset2][id][1]), to3dWorld(camera, bbox))
                if dist < bestDist:
                    bestDist = dist
                    found = True
            if not found and points > 1:
                newid = findfree(ids, False)
                predictions[camera][newid] = predictions[camera][id]
                predictions[camera][id] = [None, None, 0, False]
                predictions[camera][newid][2] = 0
                for dataset2 in cameras:
                    if camera == dataset2: continue
                    predictions[dataset2][newid] = [None, None, 0, False]
                ids.append(newid)
                id = newid

            # phase 2.2: find closest other point, if other id without point, change to that
            # TODO change to center of groups
            # TODO change only if counter >= x
            bestId = None
            bestDist = CLOSEST_DIST
            for id2 in ids[0:i + 1]:
                for dataset2 in cameras:
                    if id == id2 and camera == dataset2: continue
                    if predictions[dataset2][id2][1] is None: continue

                    dist = f_euclidian(to3dWorld(dataset2, predictions[dataset2][id2][1]), to3dWorld(camera, bbox))
                    if dist < bestDist:
                        bestId = id2
                        bestDist = dist
            if bestId is not None and predictions[camera][bestId][1] is None:
                predictions[camera][bestId] = predictions[camera][id]
                predictions[camera][id] = [None, None, 0, False]
                id = bestId

    # phase 3: set target as person if tracked continuously
    for id in ids:
        if id >= 0: continue
        person = False
        for camera in cameras:
            if predictions[camera][id][2] < -FRAMES_PERSON:
                person = True
                break
        if person:
            newid = findfree(ids, True)
            for camera in cameras:
                predictions[camera][newid] = predictions[camera][id]
                predictions[camera][id] = [None, None, 0, False]
            ids.remove(id)
            ids.append(newid)

    # phase 5: calculate dispersion of each id and remove empty ones
    newids = []
    for id in ids:
        maxdist = 0
        points = 0
        for i, camera in enumerate(cameras):
            if predictions[camera][id][1] is None: continue
            points += 1
            for dataset2 in cameras[0:i]:
                if camera == dataset2: continue
                if predictions[dataset2][id][1] is None: continue

                dist = f_euclidian(to3dWorld(camera, predictions[camera][id][1]), to3dWorld(dataset2, predictions[dataset2][id][1]))
                if dist > maxdist:
                    maxdist = dist

        # if points > 1:
        #    print "id=", id, " maxdist=", maxdist, " points=", points

        if points > 0:
            newids.append(id)
    ids = newids

    # TODO: average point, move rectangles

    return predictions, ids


def to3dWorld(dataset, bbox):
    calib = getCalibrationMatrix(dataset)
    point = Point2D(bbox.xmin + bbox.width / 2., bbox.ymax)
    return f_multiply(calib, point)


def fixbbox(frame, bbox):
    if bbox is None: return None

    height, width, colors = frame.shape

    # (0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows)
    if bbox.width <= 0 \
            or bbox.height <= 0:
        # bad bbox
        return None

    if bbox.xmin < 0:
        bbox.setXmin(0)
    if bbox.ymin < 0:
        bbox.setYmin(0)
    if bbox.xmax > width:
        bbox.setXmax(width)
    if bbox.ymax > height:
        bbox.setYmax(height)

    return bbox if bbox.isValid() else None


@cache_function("evalMultiTracker_{0}_{1}", lambda _gd, _tt, display: cache_function.TYPE_DISABLE if display else cache_function.TYPE_NORMAL, 1)
def evalMultiTracker(groupDataset, tracker_type, display=True):
    detector = {}  # detector[dataset][frame][index]

    # colors
    colors = getColors(12)

    # parse detector
    for dataset in groupDataset:
        _data = getSuperDetector(dataset)

        for iter_frames in _data.iterkeys():
            detector.setdefault(iter_frames, {})[dataset] = []
            for xmin, ymin, xmax, ymax in _data[iter_frames]:
                detector[iter_frames][dataset].append(Bbox.XmYmXMYM(xmin, ymin, xmax, ymax))

    # initialize videos
    videos = {}
    frames = {}
    for dataset in groupDataset:
        video = getVideo(dataset)
        videos[dataset] = video

        # Exit if video not opened.
        if not video.isOpened():
            print "Could not open video for dataset", dataset
            sys.exit()

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
    while allOk:

        # parse trackers
        for dataset in groupDataset:
            for id in ids:
                tracker = predictions[dataset][id][0]
                if tracker is not None:
                    # get tracker prediction
                    ok, bbox = tracker.update(frames[dataset])
                    predictions[dataset][id][1] = Bbox.XmYmWH(*bbox) if ok else None

        # detector needed
        detector_needed = frame_index % DETECTOR_FIRED == 0

        # merge all predictions -> estimations
        estimations, ids = estimateFromPredictions(predictions, ids, detector[frame_index] if detector_needed else None, groupDataset)

        # initialize new trackers
        for dataset in groupDataset:
            for id in ids:
                tracker = estimations[dataset][id][0]
                if tracker is not None: continue

                bbox = fixbbox(frames[dataset], estimations[dataset][id][1])
                estimations[dataset][id][1] = bbox

                if bbox is not None:
                    # intialize tracker
                    tracker = getTracker(tracker_type)
                    try:
                        tracker.init(frames[dataset], bbox.getAsXmYmWH())
                        estimations[dataset][id][0] = tracker
                    except BaseException:
                        print "Error on tracker init"
                else:
                    # invalid bbox, remove old tracker
                    estimations[dataset][id][0] = None

        # Show bounding boxes
        for dataset in groupDataset:
            data_detected[dataset][frame_index] = {}
            for id in ids:
                bbox = estimations[dataset][id][1]
                if bbox is None: continue

                if id >= 0:
                    data_detected[dataset][frame_index][id] = bbox.getAsXmYmXMYM()  # xmin, ymin, xmax, ymax

                # show bbox
                if display:
                    p1 = (int(bbox.xmin), int(bbox.ymin))
                    p2 = (int(bbox.xmax), int(bbox.ymax))
                    color = colors[id % len(colors)]
                    cv2.rectangle(frames[dataset], p1, p2, color, 2 if id >= 0 else 1, 1)
                    cv2.putText(frames[dataset], str(id), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.4 if id >= 0 else 0.2, color, 1)

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
                    bbox = estimations[dataset][id][1]
                    if bbox is None: continue

                    px, py = to3dWorld(dataset, bbox).getAsXY()
                    cv2.circle(frame, (int(px), int(py)), 1, colors[id % len(colors)], 2)
            cv2.imshow(WIN_NAME + "_overview", frame)
            # /end display overview

            if cv2.waitKey(1) & 0xff == 27:
                break
        else:
            # show progress
            if sys.stdout.isatty():
                sys.stdout.write("\r" + str(frame_index) + " ")
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

    return frame_index, range(max(ids) + 1), data_detected


if __name__ == '__main__':
    dataset = getGroupedDatasets()[4]
    print(getGroupedDatasets())
    tracker = getTrackers()[1]  # 0 slow good, 1 fast bad

    evalMultiTracker(dataset, tracker)
