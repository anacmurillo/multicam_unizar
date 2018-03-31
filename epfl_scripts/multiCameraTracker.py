import numpy as np
import sys

import cv2

from colorUtility import getColors
from cv2Trackers import getTracker, getTrackers
from groundTruthParser import getGroundTruth, getVideo, getGroupedDatasets

WIN_NAME = "Tracking_"


def mergeAllPredictions(previous, prediction, groupDataset):
    newPredictions = {}

    for dataset in groupDataset:
        newPredictions[dataset] = prediction[dataset]

    return newPredictions


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
                        predictions[dataset] = trackers[id][dataset][0].update(frames[dataset])

            predictions = mergeAllPredictions(trackers[id], predictions, groupDataset)

            for dataset in groupDataset:
                if predictions[dataset] is None or trackers[id][dataset] is None:
                    continue

                ok, bbox = predictions[dataset]

                if ok:
                    tracker = getTracker(tracker_type)
                    if not tracker.init(frames[dataset], bbox): print "tracker init failed"
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
    dataset = getGroupedDatasets()[4]
    print(getGroupedDatasets())
    tracker = getTrackers()[1]

    evalMultiTracker(dataset, tracker)
