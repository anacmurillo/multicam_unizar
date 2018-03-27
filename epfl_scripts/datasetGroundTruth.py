"""
Shows the groundtruth dataset.

Internal debugging utility, no real usage.
"""

import cv2  # read video file

from colorUtility import getColors
from groundTruthParser import parseFile, getFilenames

# settings
video_folder = "/home/jaguilar/Abel/epfl/dataset/CVLAB/"


def evalFile(filename):
    """
    Shows the groundtruth of the filename visually
    :param filename: the dataset filename
    """
    # read groundtruth
    track_ids, data = parseFile(filename)

    # generate colors
    persons = len(track_ids)
    print persons, "persons"
    colors_list = getColors(persons)
    colors = {}
    for i, track_id in enumerate(track_ids):
        colors[track_id] = colors_list[i]

    # read video
    vidcap = cv2.VideoCapture(video_folder + filename + ".avi")
    success, image = vidcap.read()
    if not success:
        print "invalid video"
        return
    frame = 0
    windowlabel = filename + ".jpg"
    while success:
        # draw rectangles
        for id in track_ids:
            xmin, ymin, xmax, ymax, lost, occluded, generated, label = data[frame][id]
            if not lost:
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[id], 1)

        cv2.imshow(windowlabel, image)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        success, image = vidcap.read()
        frame += 1
    if not success and frame in data:
        print len(data), "frames parsed (datafile), but", frame, "shown (video)"
    cv2.destroyWindow(windowlabel)


def runAll():
    """
    for all filenames
    """
    for filename in getFilenames():
        print filename
        evalFile(filename)


if __name__ == '__main__':
    runAll()
    # evalFile("Basketball/match5-c0")
