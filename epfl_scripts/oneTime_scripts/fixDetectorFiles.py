import csv
import numpy as np

import cv2

from epfl_scripts.Utilities.groundTruthParser import getDatasets, getVideo, getSuperDetector
from epfl_scripts.oneTime_scripts.frameExtractor import createFolder


def fixFile(dataset):
    video_good = getVideo(dataset)
    video_bad = cv2.VideoCapture("/home/jaguilar/Abel/epfl/dataset/CVLAB/" + dataset + ".avi")

    data_bad = getSuperDetector(dataset)

    filename_bad = "/home/jaguilar/Abel/epfl/dataset/superDetector/" + dataset + ".txt"
    filename_good = "/home/jaguilar/Abel/epfl/dataset/superDetector_good/" + dataset + ".txt"
    createFolder(filename_good)
    output = open(filename_good, "w")

    data_bad = {}
    reader = csv.reader(open(filename_bad), delimiter=' ')
    for frame_number, xmin, ymin, xmax, ymax in reader:
        data_bad.setdefault(int(frame_number), []).append(xmin + " " + ymin + " " + xmax + " " + ymax)


    okB, frame_bad = video_bad.read()
    index_good = -1
    index_bad = 0
    while True:
        okG, frame_good = video_good.read()
        index_good += 1

        if not okG or not okB:
            break

        while True:
            different = np.any(cv2.subtract(frame_good, frame_bad))

            if not different:
                #print index_good, "->", index_bad
                for element in data_bad.get(index_bad, []):
                    output.write(str(index_good)+" "+element+"\n")
                break
            else:
                okB, frame_bad = video_bad.read()
                if not okB:
                    break
                index_bad += 1


if __name__ == '__main__':
    # fixFile('Campus/campus7-c0')
    for dataset in getDatasets():
        print dataset, "..."
        fixFile(dataset)
