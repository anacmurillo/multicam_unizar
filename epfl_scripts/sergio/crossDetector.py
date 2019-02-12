# import cv2
import epfl_scripts.Utilities.cv2Visor as cv2
from epfl_scripts.Utilities.colorUtility import getColors, blendColors
from epfl_scripts.Utilities.geometry_utils import Bbox, f_area, f_intersection
from epfl_scripts.groundTruthParser import getGroundTruth, getVideo

OCCLUSION_IOU = 0.85
CROSS_FRAMES_BEFORE = 5
CROSS_FRAMES_DURING = 5
CROSS_FRAMES_AFTER = 5


class CrossDetector:
    class Cross:
        STATE_INVALID = -1
        STATE_DURING = 1
        STATE_AFTER = 2
        STATE_END = 3

        def __init__(self, (idFront, idBack), framesBefore, frameFirstSaw):
            self.idFront = idFront
            self.idBack = idBack
            self.frameFirstSaw = frameFirstSaw

            self.state = self.STATE_DURING if framesBefore >= CROSS_FRAMES_BEFORE else self.STATE_INVALID
            self.framesBefore = framesBefore
            self.framesDuring = 0
            self.framesAfter = 0
            self.updated = True

        def updateFrame(self, (idFront, idBack)):
            # check invalid current state
            if self.state == self.STATE_END or self.state == self.STATE_INVALID:
                # state where nothing else is needed, not used
                return False

            # check valid ids
            if idFront == self.idFront and idBack == self.idBack:
                # our ids
                if self.state == self.STATE_DURING:
                    # still a detection
                    self.framesDuring += 1
                    self.updated = True
                    return True

                if self.state == self.STATE_AFTER:
                    # new cross, treat as end
                    self.state = self.STATE_END if self.framesAfter >= CROSS_FRAMES_AFTER else self.STATE_INVALID
                    return False

            # check invalid ids
            if idFront in (idFront, idBack) or idBack in (idFront, idBack):
                # the ids conflict
                if self.state == self.STATE_AFTER and self.framesAfter >= CROSS_FRAMES_AFTER:
                    # end of cross
                    self.state = self.STATE_END
                else:
                    # invalid cross
                    self.state = self.STATE_INVALID
                return False

            # otherwise, not used
            return False

        def finishFrame(self):
            if self.updated:
                # changes made, nothing else
                self.updated = False
                return

            if self.state == self.STATE_DURING:
                # cross is no more
                if self.framesDuring >= CROSS_FRAMES_DURING:
                    # valid cross
                    self.state = self.STATE_AFTER
                    self.framesAfter += 1
                else:
                    # invalid cross
                    self.state = self.STATE_INVALID
                return

            if self.state == self.STATE_AFTER:
                self.framesAfter += 1

        def endVideo(self):
            if self.state == self.STATE_END:
                # valid
                return

            if self.state == self.STATE_AFTER and self.framesAfter >= CROSS_FRAMES_AFTER:
                # valid
                self.state = self.STATE_END
                return

            # else, invalid
            self.state = self.STATE_INVALID
            return

        def isValid(self):
            return self.state != self.STATE_INVALID

        def __str__(self):
            return "Cross {}<<{} :: {}-{}-{}-{} {}".format(self.idFront, self.idBack,
                                                           self.frameFirstSaw - self.framesBefore, self.frameFirstSaw, self.frameFirstSaw + self.framesDuring, self.frameFirstSaw + self.framesDuring + self.framesAfter,
                                                           self.state if self.state != self.STATE_END else "")

    class LastSawId:
        def __init__(self):
            self.ids = {}

        def updateFrame(self, (id1, id2), frame):
            self.ids[id1] = frame
            self.ids[id2] = frame

        def get(self, (id1, id2)):
            lastSaw = 0
            if id1 in self.ids:
                lastSaw = max(lastSaw, self.ids[id1])
            if id2 in self.ids:
                lastSaw = max(lastSaw, self.ids[id2])
            return lastSaw

    def __init__(self):
        self.crosses = []
        self.framesNotCross = self.LastSawId()

    def addOcclusion(self, occlusions, frame):
        for occlusion in occlusions:
            # update all current crosses
            used = False
            for cross in self.crosses:
                # update crosses
                used |= cross.updateFrame(occlusion)
            if not used:
                # occlusion not used, add as new cross
                self.crosses.append(self.Cross(occlusion, frame - self.framesNotCross.get(occlusion), frame))

        for occlusion in occlusions:
            # update framesNotCross
            self.framesNotCross.updateFrame(occlusion, frame)

        for cross in self.crosses:
            # end update process
            cross.finishFrame()

        for cross in self.crosses:
            print "Current :", cross

        # remove invalid
        self.crosses = [c for c in self.crosses if c.isValid()]

    def endVideo(self):
        for cross in self.crosses:
            cross.endVideo()
        self.crosses = [c for c in self.crosses if c.isValid()]

    def getCrosses(self):
        return self.crosses


def parseData((xmin, ymin, xmax, ymax, lost, occluded, generated, label)):
    """
    converts data
    :return: the bbox and 'found'
    """
    return Bbox.XmYmXMYM(xmin, ymin, xmax, ymax), not lost


def findOcclusions(track_ids, bboxes):
    """
    finds pairs of ids where
    :param track_ids: list of all possible ids
    :param bboxes: list of boxes for each id
    :return: list of pairs (id1, id2) meaning 'id2 is behind id1'
    """
    occlusions = set()

    for id1 in track_ids:
        for id2 in track_ids:
            if id2 <= id1:
                continue

            bbox1, valid1 = parseData(bboxes[id1])
            if not valid1:
                continue
            bbox2, valid2 = parseData(bboxes[id2])
            if not valid2:
                continue

            area1 = f_area(bbox1)
            area2 = f_area(bbox2)

            if area1 < area2:
                (id1, id2) = (id2, id1)
                (bbox1, bbox2) = (bbox2, bbox1)
                (area1, area2) = (area2, area1)
            # test id2 behind id1

            myiou = f_area(f_intersection(bbox1, bbox2)) / area2

            if myiou > OCCLUSION_IOU:
                occlusions.add((id1, id2))
    return occlusions


def evalOne(dataset):
    """
    Shows the groundtruth of the filename visually
    :param dataset: the dataset filename
    """
    # read groundtruth
    track_ids, data = getGroundTruth(dataset)

    # generate colors
    persons = len(track_ids)
    print persons, "persons"
    colors_list = getColors(persons)
    colors = {}
    for i, track_id in enumerate(track_ids):
        colors[track_id] = colors_list[i]

    vidcap = getVideo(dataset)
    success, image = vidcap.read()
    frame = 0
    crossDetector = CrossDetector()

    while success:
        print frame

        # draw rectangles
        for id in track_ids:
            bbox, found = parseData(data[frame][id])
            if found:
                cv2.rectangle(image, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), colors[id], 1)
                cv2.putText(image, str(id), (bbox.xmin, bbox.ymin + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[id], 1)
        cv2.putText(image, str(frame), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (1, 1, 1), 1)

        # find oclussions
        occlusions = findOcclusions(track_ids, data[frame])

        # display occlusions
        for (idA, idB) in occlusions:
            print "La persona ", idB, "pasa por detras de", idA
            bboxA, found = parseData(data[frame][idA])
            bboxB, found = parseData(data[frame][idB])

            # draw rectangles wider
            cv2.rectangle(image, (bboxA.xmin, bboxA.ymin), (bboxA.xmax, bboxA.ymax), colors[idA], 2)
            cv2.rectangle(image, (bboxB.xmin, bboxB.ymin), (bboxB.xmax, bboxB.ymax), colors[idB], 2)

            # connect rectangles
            color = blendColors(colors[idA], colors[idB])
            cv2.line(image, (bboxA.xmin, bboxA.ymin), (bboxB.xmin, bboxB.ymin), color, 2)
            cv2.line(image, (bboxA.xmin, bboxA.ymax), (bboxB.xmin, bboxB.ymax), color, 2)
            cv2.line(image, (bboxA.xmax, bboxA.ymin), (bboxB.xmax, bboxB.ymin), color, 2)
            cv2.line(image, (bboxA.xmax, bboxA.ymax), (bboxB.xmax, bboxB.ymax), color, 2)

        crossDetector.addOcclusion(occlusions, frame)

        # display image
        cv2.imshow(dataset, image)

        if cv2.waitKey(500 if len(occlusions) else 1) & 0xff == 27:
            break

        # next
        success, image = vidcap.read()
        frame += 1

    # print detections
    crossDetector.endVideo()
    for occlusion in crossDetector.getCrosses():
        print "Found Occlusion", occlusion


if __name__ == '__main__':
    evalOne('Laboratory/6p-c0')
