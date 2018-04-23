import numpy as np
import sys

import cv2

from epfl_scripts.Utilities.cv2Trackers import getTracker, getTrackers
from epfl_scripts.Utilities.groundTruthParser import getVideo, getGroupedDatasets, getSuperDetector

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
    # predictions[dataset][id] = [ tracker, (ok, bbox), framesLost, {add as neccesary} ]
    # for id in ids: etc
    # detector[dataset] = [ (bbox), ... ]
    # for dataset in groupDataset: etc

    for dataset in groupDataset:
        for id in ids:
            tracker, (ok, bbox), framesLost = predictions[dataset][id]
            if tracker is None or not ok: continue

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

            if framesLost < 10:
                predictions[dataset][id] = [tracker, (True, bestBbox), framesLost]
            else:
                predictions[dataset][id] = [None, (False, (0, 0, 0, 0)), 0]

        id = max(ids + [-1]) + 1
        for bbox in detector[dataset]:
            predictions[dataset][id] = [None, (True, bbox), 0]
            for dataset2 in groupDataset:
                if dataset == dataset2: continue
                predictions[dataset2][id] = [None, (False, (0, 0, 0, 0)), 0]
            ids.append(id)
            id += 1
    return predictions, ids


def f_subtract(pointA, pointB):
    return pointA[0] - pointB[0], pointA[1] - pointB[1], 1


def f_center(bbox):
    return bbox[0] + bbox[2] / 2., bbox[1] + bbox[3] / 2.


def homogeneous(p):
    return np.true_divide(p[0:2], p[2])


def fixbbox(frame, (ok, bbox)):
    height, width, colors = frame.shape
    newbbox = [0, 0, 0, 0]

    # (0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows)
    if not ok \
            or bbox[2] <= 0 \
            or bbox[0] + bbox[2] > width \
            or bbox[3] <= 0 \
            or bbox[1] + bbox[3] > height \
            or bbox[0] < 0 \
            or bbox[1] < 0:
        # bad bbox
        return False, newbbox

    newbbox = list(bbox)

    if bbox[0] < 0:
        newbbox[0] = 0
        newbbox[2] += bbox[0] * 2
    if bbox[1] < 0:
        newbbox[1] = 0
        newbbox[3] += bbox[1] * 2

    return True, tuple(newbbox)


def evalMultiTracker(groupDataset, tracker_type, display=True):
    detector = {}  # detector[dataset][frame][index]

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
                    predictions[dataset][id][1] = tracker.update(frames[dataset])

        # merge all predictions
        predictions, ids = mergeAllPredictions(predictions, ids, detector[frame_index], groupDataset)

        # initialize new trackers
        for dataset in groupDataset:
            for id in ids:
                ok, bbox = fixbbox(frames[dataset], predictions[dataset][id][1])

                if ok:
                    tracker = getTracker(tracker_type)
                    try:
                        tracker.init(frames[dataset], bbox)
                        predictions[dataset][id][0] = tracker
                        predictions[dataset][id][1] = (ok, bbox)
                    except BaseException:
                        print "Error on tracker init"

        # Show bounding boxes
        for dataset in groupDataset:
            data_detected[dataset][frame_index] = {}
            for id in ids:
                ok, bbox = predictions[dataset][id][1]
                if not ok: continue

                # show bbox
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                data_detected[dataset][frame_index][id] = [p1[0], p1[1], p2[0], p2[1]]  # xmin, ymin, xmax, ymax
                if display:
                    color = (255, 255, 255)
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
    tracker = getTrackers()[0]

    evalMultiTracker(dataset, tracker)
