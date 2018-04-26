import math
import numpy as np
import sys

# import cv2
import epfl_scripts.Utilities.cv2Visor as cv2
from epfl_scripts.Utilities.colorUtility import getColors
from epfl_scripts.Utilities.cv2Trackers import getTracker, getTrackers
from epfl_scripts.Utilities.groundTruthParser import getVideo, getGroupedDatasets, getSuperDetector, getCalibrationMatrix

WIN_NAME = "Tracking"


def f_iou(boxA, boxB):
    """
    IOU (Intersection over Union) of both boxes.
    :return: value in range [0,1]. 0 if disjointed bboxes, 1 if equal bboxes
    """

    boxA = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]]
    boxB = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]]

    intersection = f_area([max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])])

    union = f_area(boxA) + f_area(boxB) - intersection
    return intersection / union


def f_area(bbox):
    """
    return area of bbox
    """
    return (bbox[2] - bbox[0]) * 1. * (bbox[3] - bbox[1]) if bbox[2] > bbox[0] and bbox[3] > bbox[1] else 0.


def mergeAllPredictions(predictions, ids, detector, groupDataset):
    # predictions[dataset][id] = [ tracker, bbox, framesLost, {add as neccesary} ]
    # for id in ids: etc
    # detector[dataset] = [ (bbox), ... ]
    # for dataset in groupDataset: etc

    pos3d = []

    # update bboxes
    for dataset in groupDataset:
        for id in ids:
            tracker, bbox, framesLost = predictions[dataset][id]
            if bbox is None: continue

            # find closest detector to tracker
            bestBbox = bbox
            bestIoU = 0.5
            for detBbox in detector[dataset]:
                iou = f_iou(bbox, detBbox)
                if iou > bestIoU:
                    bestIoU = iou
                    bestBbox = detBbox

            framesLost += 1
            if bestIoU > 0.5:
                detector[dataset].remove(bestBbox)
                framesLost = 0

            # update frame
            if framesLost < 10:
                predictions[dataset][id] = [tracker, bestBbox, framesLost]
                pos3d.append((to3dWorld(dataset, bestBbox), id))
            else:
                predictions[dataset][id] = [tracker, None, 0]

    # update ids
    for dataset in groupDataset:
        for id in ids:
            tracker, bbox, framesLost = predictions[dataset][id]
            if bbox is None: continue

            # find closest other id
            bestId = None
            bestDist = 10
            for id2 in ids:
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

    # assign detections
    for dataset in groupDataset:
        for bbox in detector[dataset]:

            # find best id
            closestid = -1
            closestDist = 50
            maxid = -1
            assign = False

            for point, id in pos3d:
                dist = f_euclidian(point, to3dWorld(dataset, bbox))
                if dist < closestDist:
                    assign = True
                    closestDist = dist
                    closestid = id
                if id > maxid:
                    maxid = id

            # assign new prediction
            if not assign:
                closestid = maxid + 1
                for dataset2 in groupDataset:
                    if dataset == dataset2: continue
                    predictions[dataset2][closestid] = [None, None, 0]
                if closestid not in ids: ids.append(closestid)
            if closestid in predictions[dataset] and predictions[dataset][closestid][1] is not None:
                print "Overrided previous data"
            predictions[dataset][closestid] = [None, bbox, 0]

    # calculate dispersion of each id
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

        if points > 1:
            print "id=", id, " maxdist=", maxdist, " points=", points

    return predictions, ids


def to3dWorld(dataset, bbox):
    calib = getCalibrationMatrix(dataset)

    return homogeneous(np.dot(calib, [bbox[0] + bbox[2] / 2., bbox[1] + bbox[3], 1.]))


def f_subtract(pointA, pointB):
    return pointA[0] - pointB[0], pointA[1] - pointB[1], 1


def f_center(bbox):
    return bbox[0] + bbox[2] / 2., bbox[1] + bbox[3] / 2.


def f_euclidian(a, b):
    """
    returns the euclidian distance between the two points
    """
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def homogeneous(p):
    return np.true_divide(p[0:2], p[2])


def fixbbox(frame, bbox):
    if bbox is None: return None

    height, width, colors = frame.shape

    # (0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows)
    if bbox[2] <= 0 \
            or bbox[0] + bbox[2] > width \
            or bbox[3] <= 0 \
            or bbox[1] + bbox[3] > height \
            or bbox[0] < 0 \
            or bbox[1] < 0:
        # bad bbox
        return None

    newbbox = list(bbox)

    if bbox[0] < 0:
        newbbox[0] = 0
        newbbox[2] += bbox[0] * 2
    if bbox[1] < 0:
        newbbox[1] = 0
        newbbox[3] += bbox[1] * 2

    return tuple(newbbox)


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
                detector[iter_frames][dataset].append([xmin, ymin, xmax - xmin, ymax - ymin])

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
                    predictions[dataset][id][1] = bbox if ok else None

        # merge all predictions
        predictions, ids = mergeAllPredictions(predictions, ids, detector[frame_index], groupDataset)

        # initialize new trackers
        for dataset in groupDataset:
            for id in ids:
                bbox = fixbbox(frames[dataset], predictions[dataset][id][1])
                predictions[dataset][id][1] = bbox

                if bbox is not None:
                    # intialize tracker
                    tracker = getTracker(tracker_type)
                    try:
                        tracker.init(frames[dataset], bbox)
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

                # show bbox
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                data_detected[dataset][frame_index][id] = [p1[0], p1[1], p2[0], p2[1]]  # xmin, ymin, xmax, ymax
                if display:
                    color = colors[id % 12]
                    cv2.rectangle(frames[dataset], p1, p2, color, 2, 1)
                    cv2.putText(frames[dataset], str(id), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if display:
            # Display result
            concatenatedFrames = None
            for dataset in groupDataset:
                if concatenatedFrames is None:
                    concatenatedFrames = frames[dataset]
                else:
                    concatenatedFrames = np.hstack((concatenatedFrames, frames[dataset]))

            cv2.imshow(WIN_NAME, concatenatedFrames)
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
