"""
Shows the groundtruth dataset.

Internal debugging utility, no real usage.
"""

import cv2  # read video file

from epfl_scripts.Utilities.colorUtility import getColors
from epfl_scripts.Utilities.groundTruthParser import getGroundTruth, getDatasets

# settings
video_folder = "/home/jaguilar/Abel/epfl/dataset/CVLAB/"


def evalFile(filename):
    """
    Shows the groundtruth of the filename visually
    :param filename: the dataset filename
    """
    # read groundtruth
    track_ids, data = getGroundTruth(filename)

    # generate colors
    persons = len(track_ids)
    print persons, "persons"
    colors_list = getColors(persons)
    colors = {}
    for i, track_id in enumerate(track_ids):
        colors[track_id] = colors_list[i]

    # read video
    images = []
    vidcap = cv2.VideoCapture(video_folder + filename + ".avi")
    success, image = vidcap.read()
    frame = 0
    while success:
        # draw rectangles
        for id in track_ids:
            xmin, ymin, xmax, ymax, lost, occluded, generated, label = data[frame][id]
            if not lost:
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[id], 1)
        cv2.putText(image, str(frame), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (1, 1, 1), 1)
        images.append(image)
        success, image = vidcap.read()
        frame += 1

    frame = 0
    windowlabel = filename + ".jpg"
    while True:

        cv2.imshow(windowlabel, images[frame])
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break
        elif k == 83:
            frame += 1
        elif k == 81:
            frame -= 1
        elif k == 82:
            frame += 10
        elif k == 84:
            frame -= 10
        elif k == 80:
            frame = 0
        elif k == 87:
            frame = len(images) - 1
        elif 49 <= k <= 57:
            frame = int(len(images) * (k-48)/10.)
        else:
            print "pressed", k
        frame = max(0, min(len(images) - 1, frame))

    cv2.destroyWindow(windowlabel)


def runAll():
    """
    for all filenames
    """
    for filename in getDatasets():
        print filename
        evalFile(filename)


if __name__ == '__main__':
    # runAll()
    evalFile("Laboratory/6p-c3")
    # evaluateTracker("Basketball/match5-c0")
