import os

import cv2

from epfl_scripts.Utilities.groundTruthParser import getVideo, getDatasets

BASEFOLDER = "/home/jaguilar/Escritorio/videos/"


def createFolder(folder):
    if not os.path.exists(os.path.dirname(folder)):
        try:
            os.makedirs(os.path.dirname(folder))
        except OSError as exc:  # Guard against race condition
            import errno
            if exc.errno != errno.EEXIST:
                raise


def extract(dataset):
    video = getVideo(dataset)
    folder = BASEFOLDER + dataset + "/"

    createFolder(folder)

    index = 0

    ok, frame = video.read()
    while ok:
        cv2.imwrite(folder + str(index) + ".jpg", frame)
        index += 1
        ok, frame = video.read()
    return index


if __name__ == "__main__":
    log = open(BASEFOLDER + "datasets.txt", "w")
    for dataset in getDatasets():
        print dataset, "..."
        frames = extract(dataset)
        print "...", frames
        log.write(dataset + "," + str(frames)+"\n")
    log.close()
