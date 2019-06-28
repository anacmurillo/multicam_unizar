"""
Shows the superDetector dataset.

Internal debugging utility, no real usage.
"""

import cv2  # read video file

from epfl_scripts.Utilities import KEY
from epfl_scripts.Utilities.colorUtility import C_WHITE
from epfl_scripts.cachedDetectron import getSuperDetector
from epfl_scripts.groundTruthParser import getDatasets, getVideo

TRAIL_LENGTH = 50


def showOne(dataset):
    """
    Shows the detection of the filename visually
    :param filename: the dataset filename
    """
    # read groundtruth
    data = getSuperDetector(dataset)

    # read video
    images = []
    vidcap = getVideo(dataset)
    ok, frame = vidcap.read()
    frame_index = 0
    while ok:
        images.append(frame)
        ok, frame = vidcap.read()
        frame_index += 1

    # flags
    disp_detection = False
    disp_trail = False

    # start
    frame_index = 0
    windowlabel = dataset
    while True:
        frame_display = images[frame_index].copy()
        cv2.putText(frame_display, str(frame_index), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (1, 1, 1), 1)

        if disp_trail:
            trail_length = min(TRAIL_LENGTH, frame_index)
            for t in range(trail_length, 0, -1):
                for (xmin, ymin, xmax, ymax), mask in data[frame_index - t]:
                    cv2.circle(frame_display, (int(round((xmin + xmax) / 2)), int(round((ymin + ymax) / 2))), 1, (255, 255, 255), 2)

        if disp_detection:
            # draw rectangles
            for (xmin, ymin, xmax, ymax), mask in data[frame_index]:
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(int(round(xmin)), int(round(ymin))))
                cv2.drawContours(frame_display, contours, -1, C_WHITE, 1)
                cv2.rectangle(frame_display, (xmin, ymin), (xmax, ymax), C_WHITE, 1)

        cv2.imshow(windowlabel, frame_display)
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
            frame_index = len(images) - 1
        elif KEY.n1 <= k <= KEY.n9:
            frame_index = int(len(images) * (k - 48) / 10.)
        elif k == KEY.Q:
            disp_trail = not disp_trail
        elif k == KEY.E:
            disp_detection = not disp_detection
        else:
            print "pressed", k
        frame_index = max(0, min(len(images) - 1, frame_index))

    cv2.destroyWindow(windowlabel)


def showAll():
    """
    for all filenames
    """
    for dataset in getDatasets():
        print dataset
        showOne(dataset)


if __name__ == '__main__':
    # showAll()
    # showOne("Laboratory/6p-c3")
    showOne("Campus/campus7-c1")
