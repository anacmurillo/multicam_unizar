"""
Old version of the multicamera tracker. Keep for history reasons. Probably broken.
"""
import sys

import numpy as np

# import cv2
import epfl_scripts.Utilities.cv2Visor as cv2
from epfl_scripts.Utilities.colorUtility import getColors
from epfl_scripts.groundTruthParser import getGroundTruth, getVideo, getGroupedDatasets, getCalibrationMatrix
from epfl_scripts.trackers.cv2Trackers import getTracker

WIN_NAME = "Tracking"


def mergeAllPredictions(previous, predictions, groupDataset):
    newPredictions = {}

    average = {}
    weights = {}

    for dataset in groupDataset:
        # create average center
        if predictions[dataset] is None:
            continue

        ok, bbox = predictions[dataset]

        if not ok:
            continue

        calib = getCalibrationMatrix(dataset)

        CENTER = 0.95

        average[dataset] = homogeneous(np.dot(calib, [bbox[0] + bbox[2] / 2., bbox[1] + bbox[3] * CENTER, 1.]))
        weights[dataset] = 1 if previous[dataset][0] is not None else 1

    if len(average) > 0:
        averaged = np.average(average.values(), 0, weights.values()).tolist()
        averaged.append(1)

    for dataset in groupDataset:
        # set final prediction
        newPredictions[dataset] = predictions[dataset]

        if len(average) <= 0:
            continue

        if predictions[dataset] is None:
            bbox = [0, 0, 100, 200]
        else:
            ok, bbox = predictions[dataset]

            if not ok:
                bbox = [0, 0, 100, 200]

        center = homogeneous(np.linalg.inv(getCalibrationMatrix(dataset)).dot(averaged))

        # print "difference:", center[0] - (bbox[0] + bbox[2] / 2.), center[1] - (bbox[1] + bbox[3] * CENTER)

        newPredictions[dataset] = (True, (int(center[0] - bbox[2] / 2.), int(center[1] - bbox[3] * CENTER), bbox[2], bbox[3]))

    return newPredictions


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
    data = {}  # data[dataset][frame][track_id]
    track_ids = None

    for dataset in groupDataset:
        # parse groundtruth
        _track_ids, _data = getGroundTruth(dataset)

        if track_ids is None:
            track_ids = _track_ids
        elif track_ids != _track_ids:
            print "Invalid number of persons in different cameras!!!", track_ids, _track_ids
            return

        data[dataset] = _data

    colors_list = getColors(len(track_ids))
    trackers = {}
    for id in track_ids:
        trackers[id] = {}
        for dataset in groupDataset:
            trackers[id][dataset] = None

    # Read video
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

    frame_index = 0
    allOk = True
    while allOk:

        # parse trackers
        for id in track_ids:
            predictions = {}
            for dataset in groupDataset:
                predictions[dataset] = [False, (0, 0, 0, 0)]
                if trackers[id][dataset] is None:
                    xmin, ymin, xmax, ymax, lost, occluded, generated, label = data[dataset][frame_index][id]
                    if not lost:
                        trackers[id][dataset] = [None, False, (0, 0, 0, 0), colors_list[id]]  # tracker, found, bbox, color
                        bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
                        predictions[dataset] = [True, bbox]

                else:
                    # Update tracker
                    tracker = trackers[id][dataset][0]
                    if tracker is not None:
                        predictions[dataset] = tracker.update(frames[dataset])

            predictions = mergeAllPredictions(trackers[id], predictions, groupDataset)

            for dataset in groupDataset:
                if predictions[dataset] is None or trackers[id][dataset] is None:
                    continue

                ok, bbox = fixbbox(frames[dataset], predictions[dataset])

                if ok:
                    tracker = getTracker(tracker_type)
                    try:
                        tracker.init(frames[dataset], bbox)
                    except BaseException:
                        pass
                    trackers[id][dataset][0] = tracker

                else:
                    trackers[id][dataset][0] = None
                trackers[id][dataset][1] = ok
                trackers[id][dataset][2] = bbox

        # Show bounding boxes
        for dataset in groupDataset:
            data_detected[dataset] = {}
            data_detected[dataset][frame_index] = {}
            for id in track_ids:
                if trackers[id][dataset] is not None:
                    tracker, ok, bbox, color = trackers[id][dataset]
                    if ok:
                        # Tracking success
                        p1 = (int(bbox[0]), int(bbox[1]))
                        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                        data_detected[dataset][frame_index][id] = [p1[0], p1[1], p2[0], p2[1]]  # xmin, ymin, xmax, ymax
                        if display:
                            cv2.rectangle(frames[dataset], p1, p2, color, 2, 1)
                            cv2.putText(frames[dataset], str(id), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    else:
                        # Tracking failure
                        # cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                        pass

        if display:
            # Display result
            concatenatedFrames = None
            for dataset in groupDataset:
                if concatenatedFrames is None:
                    concatenatedFrames = frames[dataset]
                else:
                    concatenatedFrames = np.hstack((concatenatedFrames, frames[dataset]))

            cv2.imshow(WIN_NAME, concatenatedFrames)
            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                frame_index += 1
                break
        else:
            if sys.stdout.isatty():
                sys.stdout.write("\r" + str(frame_index) + " ")
                sys.stdout.flush()

        for dataset in groupDataset:
            # Read a new frame
            ok, frames[dataset] = videos[dataset].read()
            allOk = allOk and ok

        frame_index += 1

    print ""
    if display:
        cv2.destroyAllWindows()
    for dataset in groupDataset:
        videos[dataset].release()

    return frame_index, data_detected


if __name__ == '__main__':
    dataset = getGroupedDatasets()['Terrace/terrace1']
    tracker = 'BOOSTING'

    evalMultiTracker(dataset, tracker)
