"""
Returns groundtruth data and videos

normal usage:

for filename in getFilenames():
    video = getVideo(filename):
    track_ids, data = parseFile(filename)
    # do something

"""

import csv  # read coma separated value file
import os  # file operations

import cv2  # opencv

groundtruth_folder = "/home/jaguilar/Abel/epfl/dataset/merayxu-multiview-object-tracking-dataset-d2990e227c57/EPFL/"
video_folder = "/home/jaguilar/Abel/epfl/dataset/CVLAB/"


def getFilenames():
    """
    Returns list of filenames that an be used to extract groundtruth from
    :return: ["a/b","a/c",...]
    """
    filenames = []
    for path, subdirs, files in os.walk(groundtruth_folder):
        for name in files:
            filenames.append(os.path.join(os.path.relpath(path, groundtruth_folder), os.path.splitext(name)[0]))
    return filenames


def getVideo(filename):
    """
    Returns the video of the filename provided
    :param filename: the filename as returned by getFilenames()
    :return: cv2.VideoCapture
    """
    return cv2.VideoCapture(video_folder + filename + ".avi")


def parseFile(filename):
    """
    Returns the groundtruth of the filenamed provided
    :param filename: the filename as returned by getFilenames()
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

    path = groundtruth_folder + filename + ".txt"

    track_ids = set()

    # create data for each frame
    reader = csv.reader(open(path), delimiter=' ')
    data = {}
    for track_id, xmin, ymin, xmax, ymax, frame_number, lost, occluded, generated, label in reader:
        data.setdefault(int(frame_number), {})[int(track_id)] = [int(xmin), int(ymin), int(xmax), int(ymax), lost == '1', occluded == '1', generated == '1', label]
        track_ids.add(int(track_id))
    return track_ids, data
