"""
Substitutes the detectron by using a precached version on files.

Normal usage:

1) Run this file if necessary to create the cached files (will take a while)

2) Use the CachedDetectron class instead of the normal Detectron one:

detectron = CachedDetectron() if WANT_TO_USE_CACHED_DETECTRON else Detectron()

...

detectron.evaluateImage( _ , dataset + " - " + str(frameIndex) )


"""

import errno
import os
import pickle

import cv2

basefolder = "/media/datos/abel/epfl/dataset/"

superDetector_folder = basefolder + "superDetector/"


def getPath(dataset, index):
    """
    Filename for the dataset and index (foldername if index is None)
    """
    return superDetector_folder + dataset + "/" + (str(index) + ".pickle" if index is not None else "")


class CachedDetectron:
    """
    Substitutes the Detectron by loading the detections from file instead of running the detectron.
    A call to #createCachedDataset must be run first.
    """

    def __init__(self):
        self.detector = {}

    def evaluateImage(self, _, label):
        """
        The label must be in format "dataset - frameIndex", example "Laboratory/6p-c0 - 50"
        """
        dataset, frame = label.split(" - ")

        path = getPath(dataset, frame)

        if not os.path.exists(path):
            raise Exception("Cached detection wasn't found for dataset=" + dataset + " frameIndex=" + str(frame) + " (filename="+path+")")

        with open(path, 'rb') as savefile:
            data = pickle.load(savefile)
        return data


def getSuperDetector(dataset):
    """
    Returns all the detections from the "super-detector" parsing
    :param dataset: the dataset to use
    :return: dictionary with the detections where:
        keys: each of the frames ids (from 0 inclusive to video.FRAME_COUNT exclusive)
        values: list with each detection (can be empty) in the format [(xmin, ymin, xmax, ymax), mask] (where mask is the full image)
    """
    print("Loading Cached SuperDetector...")
    path = getPath(dataset, None)

    data = {}

    for filename in os.listdir(path):
        with open(path + filename, 'rb') as savefile:
            data[int(filename.split(".")[0])] = pickle.load(savefile)
    print("...done")
    return data


def createCachedDataset(dataset, detectron):
    """
    Runs the detection on the dataset and saves for later use with the Cached detectron
    """
    print("Creating cached dataset", dataset)

    video = getVideo(dataset)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video for dataset", dataset)
        return

    nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print("Cannot read video file for dataset", dataset)
        return

    # create folder
    foldername = superDetector_folder + dataset + "/"

    if not os.path.exists(os.path.dirname(foldername)):
        try:
            os.makedirs(os.path.dirname(foldername))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                return

    index = 0

    while ok:
        # get data
        data = detectron.evaluateImage(frame, "%s - %i" % (dataset, index))

        with open(foldername + str(index) + ".pickle", "wb") as savefile:
            pickle.dump(data, savefile)

        print(index, "/", nframes)

        # Read next frame.
        ok, frame = video.read()
        index += 1


if __name__ == "__main__":
    from detectron_wrapper import Detectron
    from epfl_scripts.groundTruthParser import getVideo, getDatasets

    # initialize the detector
    detectron = Detectron()

    for dataset in getDatasets():
        createCachedDataset(dataset, detectron)
