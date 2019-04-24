# import cv2
import epfl_scripts.Utilities.cv2Visor as cv2
from epfl_scripts.Utilities.colorUtility import getColors, blendColors
from epfl_scripts.Utilities.geometry2D_utils import Bbox, f_area, f_intersection
from epfl_scripts.groundTruthParser import getGroundTruth, getVideo, getGroupedDatasets

OCCLUSION_IOU = 0.75  # IOU to detect as occlusion (iff iou(intersection, bbox)>= param)
CROSS_FRAMES_BEFORE = 0  # frames required before cross
CROSS_FRAMES_DURING = 5  # frames required during cross
CROSS_FRAMES_AFTER = 0  # frames required after cross
CROSS_FRAMES_HOLE = 1  # frames between same detections to consider them the same


class Occlusion:
    def __init__(self, idFront, idBack, dataset):
        self.idFront = idFront
        self.idBack = idBack

        self.dataset = dataset

    def getIds(self):
        return sorted((self.idFront, self.idBack))

    def sameIds(self, idA, idB):
        return len({self.idFront, self.idBack, idA, idB}) == 2

    def differentIds(self, idA, idB):
        return len({self.idFront, self.idBack, idA, idB}) == 4

    def __str__(self):
        return "Occlusion in {}: {} is behind {}".format(self.dataset, self.idFront, self.idBack)


class Cross:
    """
    Saves and updates the state of a cross, defined as 'frame before/while/after/end someone crosses behind another one'
    """
    STATE_INVALID = "INVALID"
    STATE_COMPUTING = "COMPUTING"
    STATE_END = "END"

    def __init__(self, occlusion, frameBefore, frameCurrent):
        self.idA, self.idB = occlusion.getIds()

        self.state = self.STATE_COMPUTING if frameCurrent - frameBefore >= CROSS_FRAMES_BEFORE else self.STATE_INVALID
        self.frameBefore = frameBefore
        self.frameDuring = frameCurrent
        self.frameAfter = frameCurrent + 1
        self.frameEnd = -1

    def updateNotVisible(self, id, frame):
        if id not in (self.idA, self.idB):
            # not our ids
            return

        # our ids, cancel
        self._cancel(frame)

    def updateFrame(self, occlusion, frame):
        # check non update state
        if self.state != self.STATE_COMPUTING:
            # nothing else is needed, not used
            return False

        # check other ids
        if occlusion.differentIds(self.idA, self.idB):
            # other ids, just omit
            return False

        # check one more 'during'
        if occlusion.sameIds(self.idA, self.idB) and self.frameAfter + CROSS_FRAMES_HOLE >= frame:
            # next frame, update one more
            self.frameAfter = frame + 1
            return True

        # invalid, end of detection
        self._cancel(frame)
        return False

    def endVideo(self, frame):
        if self.state == self.STATE_COMPUTING:
            self._cancel(frame)

    def _cancel(self, frame):
        if self.frameAfter - self.frameDuring < CROSS_FRAMES_DURING or frame - self.frameAfter < CROSS_FRAMES_AFTER:
            # duration too short, invalid
            self.state = self.STATE_INVALID
        else:
            # duration valid, this is the end
            self.state = self.STATE_END
            self.frameEnd = frame

    def isValid(self):
        return self.state != self.STATE_INVALID

    def __str__(self):
        return "Cross between {} and {} :: {}-{}-{}-{} [{}]".format(
            self.idA, self.idB,
            self.frameBefore, self.frameDuring, self.frameAfter, self.frameEnd,
            self.state)


class CrossDetector:
    """
    Keeps possible crosses
    """

    def __init__(self):
        self.crosses = []  # list of current valid crosses
        self.lastValid = {}  # for each id, last valid (non occluded) frame, None if not visible

    def updateVisible(self, visible, frame):
        """
        Must be called BEFORE updateOcclusions
        """
        # remove not visible
        for id in self.lastValid.keys():
            if id not in visible:
                # not visible now
                del self.lastValid[id]
                for cross in self.crosses:
                    cross.updateNotVisible(id, frame)

        # update visible
        for id in visible:
            if id not in self.lastValid:
                # visible now
                self.lastValid[id] = frame

    def updateOcclusions(self, occlusions, frame):
        """
        must be called AFTER updateVisible
        """
        for occlusion in occlusions:
            # update all current crosses
            used = False
            for cross in self.crosses:
                # update crosses
                used |= cross.updateFrame(occlusion, frame)
            if not used:
                # occlusion not used, add as new possible cross
                last = max(self.lastValid[occlusion.idBack], self.lastValid[occlusion.idFront])
                self.crosses.append(Cross(occlusion, last, frame))

        for occlusion in occlusions:
            # update lastValid
            self.lastValid[occlusion.idFront] = frame + 1
            self.lastValid[occlusion.idBack] = frame + 1

        # remove invalid
        self.crosses = [c for c in self.crosses if c.isValid()]

    def endVideo(self, frame):
        # finish all current
        for cross in self.crosses:
            cross.endVideo(frame)
        self.crosses = [c for c in self.crosses if c.isValid()]

    def getCrosses(self):
        return self.crosses


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


def findOcclusionsAndVisible(track_ids, groupDataset, bboxes):
    """
    finds pairs of ids where id2 'crossed behind' id1
    :param track_ids: list of all possible ids
    :param bboxes: list of boxes for each id and dataset
    :param groupDataset: list of datasets
    :return: list of pairs (id1, id2, d) meaning 'id2 is behind id1 in dataset d'
    """
    occlusions = set()
    visible = set()

    for dataset in groupDataset:
        for id1 in track_ids:
            bbox1, valid1 = parseData(bboxes[dataset], id1)
            if not valid1:
                continue
            area1 = f_area(bbox1)
            visible.add(id1)  # added if visible

            for id2 in track_ids:
                if id2 <= id1:
                    continue

                bbox2, valid2 = parseData(bboxes[dataset], id2)
                if not valid2:
                    continue
                area2 = f_area(bbox2)

                if area1 < area2:
                    # 1 is behind 2
                    idF, idB = id2, id1
                    areaMin = area1
                else:
                    # 2 is behind 1
                    idF, idB = id1, id2
                    areaMin = area2

                myiou = f_area(f_intersection(bbox1, bbox2)) / areaMin

                if myiou > OCCLUSION_IOU:
                    occlusions.add(Occlusion(idF, idB, dataset))
    return occlusions, visible


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
    crossDetector = CrossDetector()

    while successAll:
        #print frame

        # find occlusions
        occlusions, visible = findOcclusionsAndVisible(track_ids, groupedDataset, {dataset: data[dataset][frame] for dataset in groupedDataset})

        if display:
            for occlusion in occlusions:
                print occlusion

        crossDetector.updateVisible(visible, frame)
        crossDetector.updateOcclusions(occlusions, frame)

        if display:
            for dataset in groupedDataset:
                image = images[dataset]

                # draw rectangles
                for id in track_ids:
                    bbox, found = parseData(data[dataset][frame], id)
                    if found:
                        cv2.rectangle(image, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), colors[id], 1)
                        cv2.putText(image, str(id), (bbox.xmin, bbox.ymin + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[id], 1)
                cv2.putText(image, str(frame), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (1, 1, 1), 1)

                # draw occlusions
                for occlusion in occlusions:
                    if occlusion.dataset != dataset:
                        continue
                    bboxA, found = parseData(data[dataset][frame], occlusion.idFront)
                    bboxB, found = parseData(data[dataset][frame], occlusion.idBack)

                    # draw rectangles wider
                    cv2.rectangle(image, (bboxA.xmin, bboxA.ymin), (bboxA.xmax, bboxA.ymax), colors[occlusion.idFront], 2)
                    cv2.rectangle(image, (bboxB.xmin, bboxB.ymin), (bboxB.xmax, bboxB.ymax), colors[occlusion.idBack], 2)

                    # connect rectangles
                    color = blendColors(colors[occlusion.idFront], colors[occlusion.idBack])
                    cv2.line(image, (bboxA.xmin, bboxA.ymin), (bboxB.xmin, bboxB.ymin), color, 2)
                    cv2.line(image, (bboxA.xmin, bboxA.ymax), (bboxB.xmin, bboxB.ymax), color, 2)
                    cv2.line(image, (bboxA.xmax, bboxA.ymin), (bboxB.xmax, bboxB.ymin), color, 2)
                    cv2.line(image, (bboxA.xmax, bboxA.ymax), (bboxB.xmax, bboxB.ymax), color, 2)

                # display image
                cv2.imshow(dataset, image)

            # wait if oclussions found
            if cv2.waitKey(500 if len(occlusions) else 1) & 0xff == 27:
                break

        # read new frames
        for dataset in groupedDataset:
            # Read a new frame
            success, images[dataset] = videos[dataset].read()
            successAll = successAll and success
        frame += 1

    # end
    crossDetector.endVideo(frame)
    if display:
        for dataset in groupedDataset:
            cv2.destroyWindow(dataset)

        # print detections
        for cross in crossDetector.getCrosses():
            print "Found : ", cross

    return crossDetector.getCrosses()


def saveCrosses(groupedDatasets, filename):
    with open(filename, "w") as file_out:
        file_out.write("idA,idB,frameBefore,frameDuring,frameAfter,frameEnd\n")  # manual input for readability
        for groupedDataset in groupedDatasets:
            crosses = evalOne(groupedDatasets[groupedDataset], False)
            if crosses is not None:
                file_out.write("[{}]\n".format(groupedDataset))
                for cross in crosses:
                    file_out.write(",".join(map(str, [cross.idA, cross.idB, cross.frameBefore, cross.frameDuring, cross.frameAfter, cross.frameEnd])) + "\n")


if __name__ == '__main__':
    # evalOne(['Laboratory/6p-c0'], True)
    # evalOne(getGroupedDatasets()['Laboratory/6p'], True)
    # saveCrosses({'Laboratory/6p': getGroupedDatasets()['Laboratory/6p']}, "output.txt")
    saveCrosses(getGroupedDatasets(False), "crosses0.txt")
    # for dataset in getGrDatasets():
    #    evalOne(dataset, True)
