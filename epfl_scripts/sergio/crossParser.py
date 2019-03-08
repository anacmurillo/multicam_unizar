# import cv2
from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
import numpy as np

import epfl_scripts.Utilities.cv2Visor as cv2
from epfl_scripts.groundTruthParser import getGroundTruth, getVideo, getGroupedDatasets


def getCrosses(filename):
    """
    Returns crosses found
    :param filename: name of file with crosses
    :return: type dictionary (dataset -> crosses) with:
        keys: type string, (groupedDataset elements) from the getGroupedDatasets function.
        values: type list, (list of crosses) with:
            element i: type dictionary (cross) with:
                'idA': id of one person
                'idB': id of the other person
                'frameBefore': first frame where the cross still don't happens but both persons are visible and not crossing
                'frameDuring': first frame where the cross is happening (previous frame isn't cross)
                'frameAfter': first frame where the cross is no longer, and both persons are visible and not crossing (previous frame is cross)
                'frameEnd': first frame where one person is not visible or another cross is happening (previous frame isn't cross)
    """
    crosses = {}
    currentDataset = None
    with open(filename, "r") as file_in:
        for line in file_in:
            line = line.strip()

            if line.startswith("[") and line.endswith("]"):
                # new dataset
                currentDataset = line[1:-1]
                crosses[currentDataset] = []
                continue

            if currentDataset is None:
                # still no dataset, ignore
                # print "ignoring >", line, "<"
                continue

            # valid dataset, add
            data = map(int, line.split(","))
            assert len(data) == 6
            crosses[currentDataset].append({
                'idA': data[0],
                'idB': data[1],
                'frameBefore': data[2],
                'frameDuring': data[3],
                'frameAfter': data[4],
                'frameEnd': data[5],
            })
    return crosses


def displayCross(groupedDataset, cross):
    groundTruth = {}  # groundTruth[dataset][frame][track_id]

    # get groudtruths
    for dataset in groupedDataset:
        _track_ids, _data = getGroundTruth(dataset)

        groundTruth[dataset] = _data

    # initialize videos
    images = {}
    for dataset in groupedDataset:
        video = getVideo(dataset)

        # Exit if video not opened.
        if not video.isOpened():
            print("Could not open video for dataset", dataset)
            return

        # Read all frames.
        images[dataset] = []
        video.set(cv2.CAP_PROP_POS_FRAMES, cross['frameBefore'])

        for frame in range(cross['frameBefore'], cross['frameEnd']):
            success, image = video.read()
            if not success:
                print("Error reading frame", frame, "from dataset", dataset)

            if cross['frameDuring'] <= frame < cross['frameAfter']:
                imageSub = np.ones(np.shape(image), np.uint8) * 125  # grey image
            else:
                imageSub = np.zeros(np.shape(image), np.uint8)  # black image

            if cross['idA'] in groundTruth[dataset][frame]:
                xminA, yminA, xmaxA, ymaxA, lostA, occluded, generated, label = groundTruth[dataset][frame][cross['idA']]
            else:
                lostA = True
            if cross['idB'] in groundTruth[dataset][frame]:
                xminB, yminB, xmaxB, ymaxB, lostB, occluded, generated, label = groundTruth[dataset][frame][cross['idB']]
            else:
                lostB = True

            if not lostA:
                imageSub[yminA:ymaxA, xminA:xmaxA] = image[yminA:ymaxA, xminA:xmaxA]
            if not lostB:
                imageSub[yminB:ymaxB, xminB:xmaxB] = image[yminB:ymaxB, xminB:xmaxB]

            if not lostA:
                cv2.rectangle(imageSub, (xminA, yminA), (xmaxA, ymaxA), (0, 0, 255), 1)
            if not lostB:
                cv2.rectangle(imageSub, (xminB, yminB), (xmaxB, ymaxB), (0, 255, 0), 1)

            images[dataset].append(imageSub)

    index = 0
    total = cross['frameEnd'] - cross['frameBefore']
    while True:

        # display
        for dataset in groupedDataset:
            cv2.imshow(dataset, images[dataset][index])

        # wait
        k = cv2.waitKey(0) & 0xff

        # parse key
        if k == 27:  # ESC
            break
        elif k == 83 or k == 100:  # right, d
            index += 1
        elif k == 81 or k == 97:  # left, a
            index += -1
        elif k == 82 or k == 119:  # up, w
            index += 10
        elif k == 84 or k == 115:  # down, s
            index += -10
        elif k == 80:  # start
            index = 0
        elif k == 87:  # end
            index = total - 1
        elif k != 255:  # other
            print("pressed", k)

        index = sorted([0, index, total - 1])[1]

    for dataset in groupedDataset:
        cv2.destroyWindow(dataset)


if __name__ == '__main__':
    crosses = getCrosses('crosses5.txt')
    groupedDatasets = getGroupedDatasets(False)
    # groupedDatasets = {'Laboratory/6p': getGroupedDatasets(False)['Laboratory/6p']}

    for groupedDataset in groupedDatasets:
        if groupedDataset not in crosses:
            # invalid
            print(groupedDataset, "info not available")
            continue

        for cross in crosses[groupedDataset]:
            # display each one
            print(cross)
            displayCross(groupedDatasets[groupedDataset], cross)
