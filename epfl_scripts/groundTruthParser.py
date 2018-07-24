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

basefolder = "/media/datos/abel/epfl/dataset/"

groundtruth_folder = basefolder+"merayxu-multiview-object-tracking-dataset-d2990e227c57/EPFL/"
video_folder = basefolder+"frames/"
superDetector_folder = basefolder+"superDetector/"

# download the files here: https://drive.google.com/open?id=1K02knTTWmCn00eXp2tRbwNr7BGLSsueM


def getDatasets():
    """
    Returns list of names that can be used to extract groundtruth
    :return: ["a/b","a/c",...]
    """
    datasets = []
    for path, subdirs, files in os.walk(groundtruth_folder):
        for name in files:
            datasets.append(os.path.join(os.path.relpath(path, groundtruth_folder), os.path.splitext(name)[0]))
    datasets.sort()
    return datasets


def getGroupedDatasets():
    """
    Returns list of names that can be used to extract groundtruth, grouped based on cameras from the same dataset
    :return: [["a/b","a/c",...],["b/a","b/b",...],...]
    """
    singles = getDatasets()
    multis = {}
    for single in singles:
        multi = single[0:single.rfind('-')]
        multis.setdefault(multi, []).append(single)

    del multis['Basketball/match5']
    del multis['Passageway/passageway1']
    return multis


def getVideo(dataset):
    """
    Returns the video of the dataset provided
    :param dataset: the dataset as returned by getDatasets()
    :return: cv2.VideoCapture
    """
    return cv2.VideoCapture(video_folder + dataset + "/%03d.bmp")


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


def getSuperDetector(dataset):
    """
    Return the detections from the "super-detector" parsing
    :param dataset: the dataset to use
    :return: dictionary with the detections where:
        keys: each of the frames ids (from 0 inclusive to video.FRAME_COUNT exclusive)
        values: list with each detection (can be empty) in the format [xmin, ymin, xmax, ymax]
    """
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


def getCalibrationMatrix(dataset):
    """
    Returns the calibration matrix (3x3 homography from image plane to floor plane) of the defined dataset
    :param dataset: dataset to retrieve the matrix from
    :return: the matrix
    :raise ValueError: if the dataset doesn't have a calibration matrix
    """
    if dataset in calibrationmatrixes:
        return calibrationmatrixes[dataset]
    else:
        raise ValueError("calibration matrix not parsed for dataset '" + dataset + "'")


##################### utilities ##########################

calibrationmatrixes = {
    'Passageway/passageway1-c0': [[-0.0000245975, -0.0000047863, 0.0181735812], [0.0000056945, 0.0000089998, 0.0243277264], [-0.0000000067, 0.0000006977, -0.0000552219]],
    'Passageway/passageway1-c1': [[-0.0000110292, 0.0000768559, -0.0105851797], [-0.0000097598, 0.0000196797, 0.0130811407], [0.0000000029, 0.0000004326, -0.0000362086]],
    'Passageway/passageway1-c2': [[-0.0000145114, -0.0000570495, 0.0140615401], [-0.0000033340, -0.0001351901, 0.0189803318], [-0.0000000260, -0.0000003176, 0.0000289364]],
    'Passageway/passageway1-c3': [[0.0013816829, 0.0001111826, -0.1471471590], [0.0004031272, 0.0153807950, -2.8417419736], [0.0000000011, 0.0000330623, -0.0031355590]],

    'Terrace/terrace1-c0': [[-1.6688907435, -6.9502305710, 940.69592392565], [1.1984806153, -10.7495778320, 868.29873467315], [0.0004069210, -0.0209324057, 0.42949125235]],
    'Terrace/terrace1-c1': [[0.6174778372, -0.4836875683, 147.00510919005], [0.5798503075, 3.8204849039, -386.096405131], [0.0000000001, 0.0077222239, -0.01593391935]],
    'Terrace/terrace1-c2': [[-0.2717592338, 1.0286363982, -17.6643219215], [-0.1373600672, -0.3326731339, 161.0109069274], [0.0000600052, 0.0030858398, -0.04195162855]],
    'Terrace/terrace1-c3': [[-0.3286861858, 0.1142963200, 130.25528281945], [0.1809954834, -0.2059386455, 125.0260427323], [0.0000693641, 0.0040168154, -0.08284534995]],

    'Laboratory/6p-c0': [[0.176138, 0.647589, -63.412272], [-0.180912, 0.622446, -0.125533], [-0.000002, 0.001756, 0.102316]],
    'Laboratory/6p-c1': [[0.177291, 0.004724, 31.224545], [0.169895, 0.661935, -79.781865], [-0.000028, 0.001888, 0.054634]],
    'Laboratory/6p-c2': [[-0.104843, 0.099275, 50.734500], [0.107082, 0.102216, 7.822562], [-0.000054, 0.001922, -0.068053]],
    'Laboratory/6p-c3': [[-0.142865, 0.553150, -17.395045], [-0.125726, 0.039770, 75.937144], [-0.000011, 0.001780, 0.015675]],

    'Campus/campus7-c0': [[-0.211332, -0.405226, 70.781223], [-0.019746, -1.564936, 226.377280], [-0.000025, -0.001961, 0.160791]],
    'Campus/campus7-c1': [[0.000745, 0.350335, -98.376103], [-0.164871, -0.390422, 54.081423], [0.000021, -0.001668, 0.111075]],
    'Campus/campus7-c2': [[0.089976, 1.066795, -152.055667], [-0.116343, 0.861342, -75.122116], [0.000015, 0.001442, -0.064065]],

}


# javascript to extract from files:
# s=prompt("message","")
# s=s.split(/\s+/g)
# prompt("result", array([array(s.slice(0,3)),array(s.slice(3,6)),array(s.slice(6,9))]))
#
# function array(a){
#   return "["+a[0]+","+a[1]+","+a[2]+"]";
# }


def s2f2i(string):
    """
    Converts a string to an int parsing to a float first (otherwise "*.0" numbers make an error)
    string -> float -> round -> int

    "1.0"  ->  1
    "5.5"  ->  6
    "-1.0" -> -1
    "-5.5" -> -6

    :param string: string
    :return: int
    """
    return int(round(float(string)))


if __name__ == "__main__":
    print "\n- ".join(["Datasets available:"] + getDatasets())
    print
    print "\n- ".join(["Grouped datasets available:"] + getGroupedDatasets().keys())
    print
