"""
Shows the detector, multitracker and groundtruth all-in-one

Internal debugging utility, no real usage.
"""
import os
import sys

import cv2  # read video file

from epfl_scripts.Utilities import KEY
from epfl_scripts.Utilities.colorUtility import getColors, blendColors, C_WHITE, C_BLACK
from epfl_scripts.Utilities.geometry2D_utils import f_iou, Bbox, f_intersection
from epfl_scripts.cachedDetectron import getSuperDetector
from epfl_scripts.groundTruthParser import getDatasets, getVideo, getGroupedDatasets, getGroundTruth
from epfl_scripts.multiCameraTrackerV2 import evalMultiTracker


def showOne(groupDataset, tracker, filename=None, flags="0000"):
    """
    Shows the cameras of the :param groupDataset: and :param tracker: with ability to show/hide the following:
    -detector regions
    -multitracker results
    -groundtruth regions
    -bestIOU region
    """
    data_detector = {}
    data_groundTruth = {}
    vidcap = {}

    groundtruth_ids = set()

    if filename is not None:
        print "saving", str(groupDataset), "with flags", flags, "to file", filename

    for dataset in groupDataset:
        # read detector
        data_detector[dataset] = getSuperDetector(dataset)

        # read groundtruth
        gt_ids, data_groundTruth[dataset] = getGroundTruth(dataset)
        groundtruth_ids.update(gt_ids)

        # read videos
        vidcap[dataset] = getVideo(dataset)

        if filename is not None:
            vidcap[dataset + '_out'] = cv2.VideoWriter(
                filename
                + "".join(x if x.isalnum() else "_" for x in dataset)
                + "_"
                + flags
                + '.avi', cv2.VideoWriter_fourcc(*'XVID'), 25.0, (int(vidcap[dataset].get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap[dataset].get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # read algorithm output
    nFrames, trIds, data_tracker = evalMultiTracker(groupDataset, tracker, False, 10)

    # list colors
    colors_tracker = dict(zip(trIds, [blendColors(c, C_WHITE, 0.5) for c in getColors(len(trIds))]))
    colors_groundtruth = dict(zip(groundtruth_ids, [blendColors(c, C_BLACK, 0.5) for c in getColors(len(groundtruth_ids))]))

    # flags
    disp_detector = flags[0] == "1"
    disp_tracker = flags[1] == "1"
    disp_groundtruth = flags[2] == "1"
    disp_areas = flags[3] == "1"

    # start
    frame_index = 0
    while True:

        for dataset in groupDataset:

            # read frame_index frame
            vidcap[dataset].set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_display = vidcap[dataset].read()
            if not ok:
                print "Invalid frame", frame_index, "from dataset", dataset
                return

            # set index number
            cv2.putText(frame_display, str(frame_index), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_BLACK, 1)

            # draw detector
            if disp_detector:
                for (xmin, ymin, xmax, ymax), mask in data_detector[dataset][frame_index]:
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(int(round(xmin)), int(round(ymin))))
                    cv2.drawContours(frame_display, contours, -1, C_WHITE, 1)
                    cv2.rectangle(frame_display, (xmin, ymin), (xmax, ymax), C_WHITE, 1)

            # draw tracker
            if disp_tracker:
                for id in data_tracker[dataset][frame_index]:
                    xmin, ymin, xmax, ymax = data_tracker[dataset][frame_index][id]
                    cv2.rectangle(frame_display, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colors_tracker[id], 1)
                    cv2.putText(frame_display, str(id), (int(xmin), int(ymin) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors_tracker[id], 1)

            # draw groundtruth
            if disp_groundtruth:
                for id in data_groundTruth[dataset][frame_index]:
                    xmin, ymin, xmax, ymax, lost, _, _, _ = data_groundTruth[dataset][frame_index][id]
                    if lost: continue
                    cv2.rectangle(frame_display, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colors_groundtruth[id], 1)
                    cv2.putText(frame_display, str(id), (int(xmin), int(ymin) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors_groundtruth[id], 1)

            # draw areas
            if disp_areas:
                for id in data_tracker[dataset][frame_index]:
                    xmin, ymin, xmax, ymax = data_tracker[dataset][frame_index][id]
                    bboxTR = Bbox.XmYmXMYM(xmin, ymin, xmax, ymax)

                    # find best IOU
                    bestIOU = 0
                    bestArea = None
                    bestColor = C_BLACK
                    for idGT in data_groundTruth[dataset][frame_index]:
                        xmin, ymin, xmax, ymax, lost, _, _, _ = data_groundTruth[dataset][frame_index][idGT]
                        if lost: continue
                        bboxGT = Bbox.XmYmXMYM(xmin, ymin, xmax, ymax)

                        iou = f_iou(bboxTR, bboxGT)
                        if iou > bestIOU:
                            bestIOU = iou
                            bestArea = f_intersection(bboxGT, bboxTR)
                            bestColor = colors_groundtruth[idGT]

                    if bestIOU > 0:
                        transparent(bestIOU, cv2.rectangle, frame_display, (int(bestArea.xmin), int(bestArea.ymin)), (int(bestArea.xmax), int(bestArea.ymax)), blendColors(bestColor, colors_tracker[id], 0.5), cv2.FILLED)

            # show
            if filename is not None:
                vidcap[dataset + '_out'].write(frame_display)
                sys.stdout.write("\r" + str(frame_index) + "/" + str(nFrames) + "          ")
            else:
                cv2.imshow(dataset, frame_display)

        if filename is not None:
            frame_index += 1
            if frame_index == nFrames:
                break
        else:
            k = cv2.waitKey(0) & 0xff
            if k == KEY.ESC:
                break
            elif k == KEY.RIGHT_ARROW or k == KEY.D:
                frame_index += 1
            elif k == KEY.LEFT_ARROW or k == KEY.A:
                frame_index -= 1
            elif k == KEY.UP_ARROW or k == KEY.W:
                frame_index += 10
            elif k == KEY.DOWN_ARROW or k == KEY.S:
                frame_index -= 10
            elif k == KEY.START:
                frame_index = 0
            elif k == KEY.END:
                frame_index = nFrames - 1
            elif KEY.n1 <= k <= KEY.n9:
                frame_index = int(nFrames * (k - 48) / 10.)

            elif k == KEY.F1:
                disp_detector = not disp_detector
            elif k == KEY.F2:
                disp_tracker = not disp_tracker
            elif k == KEY.F3:
                disp_groundtruth = not disp_groundtruth
            elif k == KEY.F4:
                disp_areas = not disp_areas

            else:
                print "pressed", k
            frame_index = max(0, min(nFrames - 1, frame_index))

    if filename is not None:
        for dataset in groupDataset:
            vidcap[dataset + '_out'].release()
    else:
        cv2.destroyAllWindows()

    for dataset in groupDataset:
        vidcap[dataset].release()


def transparent(alpha, function, *var, **vars):
    """
    Applies cv2 :param function: with transparency :param alpha:
    """
    original = var[0].copy()
    function(*var, **vars)
    cv2.addWeighted(var[0], alpha, original, 1 - alpha, 0, var[0])


def showAll():
    """
    for all filenames
    """
    for dataset in getDatasets():
        print dataset
        showOne(dataset, 'KCF')


def saveAll():
    for flags in ["0000", "1000", "0100", "0010", "0001"]:
        for groupDataset in getGroupedDatasets().values():
            for dataset in groupDataset:
                showOne([dataset], 'KCF', 'videos/', flags)


if __name__ == '__main__':
    os.chdir("..")  # to be on the directory with _cache_

    # showAll()
    # saveAll()

    tracker = 'KCF'
    dataset = getGroupedDatasets()["Laboratory/6p"]
    # dataset = ["Laboratory/6p-c3"]
    # dataset = ["Campus/campus7-c1"]
    showOne(dataset, tracker)

    # showOne(getGroupedDatasets()["Campus/campus7"], 'KCF', 'videos/10_', "0100")
