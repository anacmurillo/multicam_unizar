"""
Returns groundtruth data and videos

normal usage:

for dataset in getDatasets():
    video = getVideo(dataset):
    track_ids, data = getGroundTruth(dataset)
    # do something

"""

import csv  # read coma separated value file
import os  # file operations

import cv2  # opencv

groundtruth_folder = "/home/jaguilar/Abel/epfl/dataset/merayxu-multiview-object-tracking-dataset-d2990e227c57/EPFL/"
video_folder = "/home/jaguilar/Abel/epfl/dataset/CVLAB/"
superDetector_folder = "/home/jaguilar/Abel/epfl/dataset/superDetector/"


def getDatasets():
    """
    Returns list of names that can be used to extract groundtruth
    :return: ["a/b","a/c",...]
    """
    datasets = []
    for path, subdirs, files in os.walk(groundtruth_folder):
        for name in files:
            datasets.append(os.path.join(os.path.relpath(path, groundtruth_folder), os.path.splitext(name)[0]))
    return datasets


def getGroupedDatasets():
    singles = getDatasets()
    multis = {}
    for single in singles:
        multi = single[0:single.rfind('-') + 1]
        multis.setdefault(multi, []).append(single)
    return multis.values()


def getVideo(dataset):
    """
    Returns the video of the dataset provided
    :param dataset: the dataset as returned by getDatasets()
    :return: cv2.VideoCapture
    """
    return cv2.VideoCapture(video_folder + dataset + ".avi")


def getGroundTruth(dataset):
    """
    Returns the groundtruth of the dataset provided
    :param dataset: the dataset as returned by getDatasets()
    :return: (tracks_ids, data) where:
        track_ids: list of ids, each id is one person (example [0,1,2,3])
        data: dictionary, frame->frame_info
            -keys: each frame number
            -values: dictionary, track_id->data
                -keys: each track_id from track_ids
                -values: list with the folowing information: xmin, ymin, xmax, ymax, lost, occluded, generated, label
                    xmin: The top left x-coordinate of the bounding box.
                    ymin: The top left y-coordinate of the bounding box.
                    xmax: The bottom right x-coordinate of the bounding box.
                    ymax: The bottom right y-coordinate of the bounding box.
                    lost: If True, the annotation is outside of the view screen.
                    occluded: If True, the annotation is occluded.
                    generated: If True, the annotation was automatically interpolated.
                    label: human, car/vehicle, bicycle.
    (example {0: {0: [0,0,1,1,False,False,False,'human'], 1: [0,0,1,1,False,False,False,'human']}, 1: {0: [0,0,1,1,False,False,False,'human'], 1: [0,0,1,1,False,False,False,'human']}})

    """

    path = groundtruth_folder + dataset + ".txt"

    track_ids = set()

    # create data for each frame
    reader = csv.reader(open(path), delimiter=' ')
    data = {}
    for track_id, xmin, ymin, xmax, ymax, frame_number, lost, occluded, generated, label in reader:
        data.setdefault(int(frame_number), {})[int(track_id)] = [int(xmin), int(ymin), int(xmax), int(ymax), lost == '1', occluded == '1', generated == '1', label]
        track_ids.add(int(track_id))
    return track_ids, data


def getCalibrationMatrix(dataset):
    if dataset == 'Laboratory/6p-c0':
        return [[0.176138, 0.647589, -63.412272], [-0.180912, 0.622446, -0.125533], [-0.000002, 0.001756, 0.102316]]
    elif dataset == 'Laboratory/6p-c1':
        return [[0.177291, 0.004724, 31.224545], [0.169895, 0.661935, -79.781865], [-0.000028, 0.001888, 0.054634]]
    elif dataset == 'Laboratory/6p-c2':
        return [[-0.104843, 0.099275, 50.734500], [0.107082, 0.102216, 7.822562], [-0.000054, 0.001922, -0.068053]]
    elif dataset == 'Laboratory/6p-c3':
        return [[-0.142865, 0.553150, -17.395045], [-0.125726, 0.039770, 75.937144], [-0.000011, 0.001780, 0.015675]]
    else:
        raise ValueError("calibration matrix not parsed for dataset '"+dataset+"'")


def getSuperDetector(dataset):
    path = superDetector_folder + dataset + ".txt"

    # create data for each frame
    reader = csv.reader(open(path), delimiter=' ')
    data = {}
    for frame_number, xmin, ymin, xmax, ymax in reader:
        data.setdefault(int(frame_number), []).append([s2f2i(xmin), s2f2i(ymin), s2f2i(xmax), s2f2i(ymax)])
    for i in range(int(getVideo(dataset).get(cv2.CAP_PROP_FRAME_COUNT))):
        if i not in data:
            data[i] = []
    return data


def s2f2i(f):
    return int(round(float(f)))


if __name__ == "__main__":
    print getGroupedDatasets()
