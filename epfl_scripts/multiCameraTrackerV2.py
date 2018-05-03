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
FRAMES_LOST = 5  # maximum number of frames until the detection is removed

FRAMES_PERSON = 10  # after this number of frames with at least one camera tracking a target (id<0), it is assigned as person (id>=0)

CLOSEST_DIST = 20  # if an existing point is closer than this to a different point id, it is assigned that same id
FARTHEST_DIST = 50  # if an existing point is farther than the rest of the group, it is removed
DETECTION_DIST = 50  # if a new point is closer than this to an existing point, it is assigned the same id


def findfree(group, person):
    free = 0 if person else -1
    while free in group:
        free += 1 if person else -1
    return free


def mergeAllPredictions(predictions, ids, detector, groupDataset):
    # predictions[dataset][id] = [ tracker, bbox, framesLost, {add as necessary} ]
    # for id in ids: etc
    # detector[dataset] = [ (bbox), ... ]
    # for dataset in groupDataset: etc

    pos3d = []

    # update bboxes
    for dataset in groupDataset:
        detectorUsed = []
        for id in ids:
            tracker, bbox, framesLost = predictions[dataset][id]
            if bbox is None: continue

            # find closest detector to tracker
            bestBbox = bbox
            bestIoU = IOU_THRESHOLD
            for detBbox in detector[dataset] + detectorUsed if detector is not None else []:
                iou = f_iou(bbox, detBbox)
                if iou > bestIoU:
                    bestIoU = iou
                    bestBbox = detBbox

            if bestIoU > IOU_THRESHOLD:
                # prediction found, remove from detections but keep for others
                if bestBbox in detector[dataset]:
                    detector[dataset].remove(bestBbox)
                    detectorUsed.append(bestBbox)
                framesLost = min(0, framesLost - 1)
            elif detector is not None:
                # not found, lost if detector was active
                framesLost = max(0, framesLost + 1)

            # update frame
            if framesLost < FRAMES_LOST:
                predictions[dataset][id] = [tracker if bestBbox == bbox else None, bestBbox, framesLost]
                pos3d.append((to3dWorld(dataset, bestBbox), id))
            else:
                predictions[dataset][id] = [None, None, 0]

    # update ids individually
    for i, id in enumerate(ids[:]):
        for dataset in groupDataset:
            tracker, bbox, framesLost = predictions[dataset][id]
            if bbox is None: continue

            # find closest other id
            bestId = None
            bestDist = CLOSEST_DIST
            for id2 in ids[0:i]:
                if predictions[dataset][id2][1] is not None: continue
                for dataset2 in groupDataset:
                    if id == id2 and dataset == dataset2: continue
                    if predictions[dataset2][id2][1] is None: continue

                    dist = f_euclidian(to3dWorld(dataset2, predictions[dataset2][id2][1]), to3dWorld(dataset, bbox))
                    if dist < bestDist:
                        bestId = id2
                        bestDist = dist
            if bestId is not None:
                predictions[dataset][bestId] = predictions[dataset][id]
                predictions[dataset][id] = [None, None, 0]
                id = bestId

            # find closest id in group
            bestDist = FARTHEST_DIST
            found = False
            points = 1
            for dataset2 in groupDataset:
                if dataset2 == dataset: continue
                if predictions[dataset2][id][1] is None: continue
                points += 1
                dist = f_euclidian(to3dWorld(dataset2, predictions[dataset2][id][1]), to3dWorld(dataset, bbox))
                if dist < bestDist:
                    bestDist = dist
                    found = True
            if not found and points > 1:
                newid = findfree(ids, False)
                predictions[dataset][newid] = predictions[dataset][id]
                predictions[dataset][id] = [None, None, 0]
                for dataset2 in groupDataset:
                    if dataset == dataset2: continue
                    predictions[dataset2][newid] = [None, None, 0]
                ids.append(newid)
                id = newid

    # set target as person if tracked continuously
    for id in ids:
        if id >= 0: continue
        person = False
        for dataset in groupDataset:
            if predictions[dataset][id][2] < -FRAMES_PERSON:
                person = True
                break
        if person:
            newid = findfree(ids, True)
            for dataset in groupDataset:
                predictions[dataset][newid] = predictions[dataset][id]
                predictions[dataset][id] = [None, None, 0]
            ids.remove(id)
            ids.append(newid)

    # assign detections
    if detector is not None:
        for dataset in groupDataset:
            for bbox in detector[dataset]:

                # find best id
                closestid = None
                closestDist = DETECTION_DIST

                for point, id in pos3d:
                    dist = f_euclidian(point, to3dWorld(dataset, bbox))
                    if dist < closestDist:
                        closestDist = dist
                        closestid = id

                # assign new prediction
                if closestid is None:
                    closestid = findfree(ids, False)
                    for dataset2 in groupDataset:
                        predictions[dataset2][closestid] = [None, None, 0]
                    ids.append(closestid)
                # if closestid in predictions[dataset] and predictions[dataset][closestid][1] is not None:
                #    print "Overrided previous data"
                predictions[dataset][closestid] = [None, bbox, 0]

    # calculate dispersion of each id and remove empty ones
    newids = []
    for id in ids:
        maxdist = 0
        points = 0
        for i, dataset in enumerate(groupDataset):
            if predictions[dataset][id][1] is None: continue
            points += 1
            for dataset2 in groupDataset[0:i]:
                if dataset == dataset2: continue
                if predictions[dataset2][id][1] is None: continue

                dist = f_euclidian(to3dWorld(dataset, predictions[dataset][id][1]), to3dWorld(dataset2, predictions[dataset2][id][1]))
                if dist > maxdist:
                    maxdist = dist

        # if points > 1:
        #    print "id=", id, " maxdist=", maxdist, " points=", points

        if points > 0:
            newids.append(id)
    ids = newids

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


@cache_function("evalMultiTracker_{0}_{1}", lambda _gd, _tt, display: display)
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

        # merge all predictions
        predictions, ids = mergeAllPredictions(predictions, ids, detector[frame_index] if detector_needed else None, groupDataset)

        # initialize new trackers
        for dataset in groupDataset:
            for id in ids:
                tracker = predictions[dataset][id][0]
                if tracker is not None: continue

                bbox = fixbbox(frames[dataset], predictions[dataset][id][1])
                predictions[dataset][id][1] = bbox

                if bbox is not None:
                    # intialize tracker
                    tracker = getTracker(tracker_type)
                    try:
                        tracker.init(frames[dataset], bbox.getAsXmYmWH())
                        predictions[dataset][id][0] = tracker
                    except BaseException:
                        print "Error on tracker init"
                else:
                    # invalid bbox, remove old tracker
                    predictions[dataset][id][0] = None

        # Show bounding boxes
        for dataset in groupDataset:
            data_detected[dataset][frame_index] = {}
            for id in ids:
                bbox = predictions[dataset][id][1]
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
                    bbox = predictions[dataset][id][1]
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

        frame_index += 1

    # clean
    print ""
    if display:
        cv2.destroyWindow(WIN_NAME)

    for dataset in groupDataset:
        videos[dataset].release()

    return frame_index, data_detected


if __name__ == '__main__':
    dataset = getGroupedDatasets()[4]
    print(getGroupedDatasets())
    tracker = getTrackers()[1]  # 0 slow good, 1 fast bad

    evalMultiTracker(dataset, tracker)
