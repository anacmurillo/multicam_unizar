"""
Shows the superDetector dataset.

Internal debugging utility, no real usage.
"""

import cv2  # read video file

from epfl_scripts.Utilities.groundTruthParser import getDatasets, getSuperDetector, getVideo


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
        # draw rectangles
        for xmin, ymin, xmax, ymax in data[frame_index]:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
        cv2.putText(frame, str(frame_index), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (1, 1, 1), 1)
        images.append(frame)
        ok, frame = vidcap.read()
        frame_index += 1

    frame_index = 0
    windowlabel = dataset
    while True:

        cv2.imshow(windowlabel, images[frame_index])
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break
        elif k == 83:
            frame_index += 1
        elif k == 81:
            frame_index -= 1
        elif k == 82:
            frame_index += 10
        elif k == 84:
            frame_index -= 10
        elif k == 80:
            frame_index = 0
        elif k == 87:
            frame_index = len(images) - 1
        elif 49 <= k <= 57:
            frame_index = int(len(images) * (k - 48) / 10.)
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
