# import cv2
import epfl_scripts.Utilities.cv2Visor as cv2
from epfl_scripts.Utilities.colorUtility import getColors, blendColors
from epfl_scripts.Utilities.geometry_utils import Bbox, f_area, f_intersection
from epfl_scripts.groundTruthParser import getGroundTruth, getVideo, getGroupedDatasets


class Person:
    # current status of a person
    def __init__(self):
        # not visible and never saw
        self.saw = -1
        self.visible = False

    def setVisible(self, visible):
        if self.visible == visible:
            # same visibility state, nothing
            return -1
        elif visible:
            # now is visible, wasn't previously
            self.visible = True
            self.saw += 1
            return self.saw
        else:
            # now is invisible, was previosuly
            self.visible = False
            return -1


def parseData(data, id):
    """
    converts data
    :return: the bbox and 'found'
    """
    if id in data:
        xmin, ymin, xmax, ymax, lost, occluded, generated, label = data[id]
        return Bbox.XmYmXMYM(xmin, ymin, xmax, ymax), not lost
    else:
        return None, False


def evalOne(groupedDataset, display):
    """
    Finds the cross on this video
    :param dataset: the dataset filename
    :param display:
    """
    data = {}  # data[dataset][frame][track_id]
    track_ids = set()

    # get groudtruths
    for dataset in groupedDataset:
        _track_ids, _data = getGroundTruth(dataset)

        track_ids.update(_track_ids)
        data[dataset] = _data

    # generate colors
    persons = len(track_ids)
    print persons, "persons"
    colors_list = getColors(persons)
    colors = {}
    for i, track_id in enumerate(track_ids):
        colors[track_id] = colors_list[i]

    # initialize videos
    videos = {}
    images = {}
    successAll = True
    for dataset in groupedDataset:
        video = getVideo(dataset)
        videos[dataset] = video

        # Exit if video not opened.
        if not video.isOpened():
            print "Could not open video for dataset", dataset
            return

        # Read first frame.
        success, image = video.read()
        images[dataset] = image
        successAll = success and successAll

    # initialize other
    frame = 0
    saws = []
    persons = {}
    for id in track_ids:
        persons[id] = Person()

    while successAll:
        print frame

        # find new saws
        newsaws = {}
        for id in track_ids:
            visible = False
            for dataset in groupedDataset:
                visible |= parseData(data[dataset][frame], id)[1]
            saw = persons[id].setVisible(visible)

            if saw != -1:
                # new event, to the list
                newsaws[id]= saw
                saws.append((id, saw, frame))

        if display:
            for dataset in groupedDataset:
                image = images[dataset]

                # draw rectangles
                for id in track_ids:
                    bbox, found = parseData(data[dataset][frame], id)
                    if found:
                        cv2.rectangle(image, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), colors[id], 3 if id in newsaws else 1)
                        cv2.putText(image, str(id), (bbox.xmin, bbox.ymin + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[id], 1)

                        # draw if new
                        if id in newsaws:
                            cv2.putText(image, str(newsaws[id]), (bbox.xmin, bbox.ymax), cv2.FONT_HERSHEY_SIMPLEX, 3, colors[id], 1)

                cv2.putText(image, str(frame), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (1, 1, 1), 1)

                # display image
                cv2.imshow(dataset, image)

            # wait if newpersons found
            if cv2.waitKey(1000 if len(newsaws) else 1) & 0xff == 27:
                break

        # read new frames
        for dataset in groupedDataset:
            # Read a new frame
            success, images[dataset] = videos[dataset].read()
            successAll = successAll and success
        frame += 1

    # end
    if display:
        for dataset in groupedDataset:
            cv2.destroyWindow(dataset)

        # print newids
        for el in saws:
            print "Found: ", el

    return saws


def saveSaws(groupedDatasets, filename):
    with open(filename, "w") as file_out:
        file_out.write("id,sawTime,frame\n")  # manual input for readability
        for groupedDataset in groupedDatasets:
            saws = evalOne(groupedDatasets[groupedDataset], False)
            if saws is not None:
                file_out.write("[{}]\n".format(groupedDataset))
                for id, sawTime, frame in saws:
                    file_out.write(",".join(map(str, [id, sawTime, frame])) + "\n")


if __name__ == '__main__':
    # evalOne(['Laboratory/6p-c0'], True)
    # print evalOne(getGroupedDatasets()['Laboratory/6p'], False)
    saveSaws(getGroupedDatasets(False), "newDetections.txt")
    # for dataset in getGroupedDatasets(False).values():
    #    evalOne(dataset, True)
