# import cv2
import epfl_scripts.Utilities.cv2Visor as cv2
from epfl_scripts.Utilities.colorUtility import getColors, blendColors
from epfl_scripts.Utilities.geometry_utils import Bbox, f_area, f_intersection
from epfl_scripts.groundTruthParser import getGroundTruth, getVideo

OCCLUSION_IOU = 0.85  # IOU to detect as occlusion (iff iou(intersection, bbox)>= param)
CROSS_FRAMES_BEFORE = 5  # frames required before cross
CROSS_FRAMES_DURING = 5  # frames required during cross
CROSS_FRAMES_AFTER = 5  # frames required after cross


class CrossDetector:
    class Cross:
        """
        Saves and updates the state of a cross, defined as 'frame before/while/after/end someone crosses behind another one'
        """
        STATE_INVALID = "INVALID"
        STATE_COMPUTING = "COMPUTING"
        STATE_END = "END"

        def __init__(self, (idFront, idBack), frameBefore, frameCurrent):
            self.idFront = idFront
            self.idBack = idBack

            self.state = self.STATE_COMPUTING if frameCurrent - frameBefore >= CROSS_FRAMES_BEFORE else self.STATE_INVALID
            self.frameBefore = frameBefore
            self.frameDuring = frameCurrent
            self.frameAfter = frameCurrent + 1
            self.frameEnd = -1

        def updateFrame(self, (idFront, idBack), frame):
            # check non update state
            if self.state != self.STATE_COMPUTING:
                # nothing else is needed, not used
                return False

            # check other ids
            if len({self.idFront, self.idBack, idFront, idBack}) == 4:
                # other ids, just omit
                return False

            # check one more 'during'
            if idFront == self.idFront and idBack == self.idBack and self.frameAfter == frame:
                # next frame, update one more
                self.frameAfter = frame + 1
                return True

            # invalid, end of detection
            if self.frameAfter - self.frameDuring < CROSS_FRAMES_DURING or frame - self.frameAfter < CROSS_FRAMES_AFTER:
                # duration too short, invalid
                self.state = self.STATE_INVALID
            else:
                # duration valid, this is the end
                self.state = self.STATE_END
                self.frameEnd = frame
            return False

        def isValid(self):
            return self.state != self.STATE_INVALID

        def __str__(self):
            return "Cross {} behind {} :: {}-{}-{}-{} [{}]".format(
                self.idBack, self.idFront,
                self.frameBefore, self.frameDuring, self.frameAfter, self.frameEnd,
                self.state)

    def __init__(self):
        self.crosses = []
        self.lastSaw = {}

    def addOcclusions(self, occlusions, frame):
        for (idF, idB) in occlusions:
            # update all current crosses
            used = False
            for cross in self.crosses:
                # update crosses
                used |= cross.updateFrame((idF, idB), frame)
            if not used:
                # occlusion not used, add as new cross
                last = 0
                if idF in self.lastSaw:
                    last = max(last, self.lastSaw[idF])
                if idB in self.lastSaw:
                    last = max(last, self.lastSaw[idB])
                self.crosses.append(self.Cross((idF, idB), last, frame))

        for (idF, idB) in occlusions:
            # update framesNotCross
            self.lastSaw[idF] = frame
            self.lastSaw[idB] = frame

        for cross in self.crosses:
            print "Current :", cross

        # remove invalid
        self.crosses = [c for c in self.crosses if c.isValid()]

    def endVideo(self, frame):
        for cross in self.crosses:
            cross.updateFrame((-1, -1), frame)  # forces end of all detections
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
    finds pairs of ids where id2 'crossed behind' id1
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


def evalOne(dataset, display):
    """
    Finds the cross on this video
    :param dataset: the dataset filename
    :param display:
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

        # find oclussions
        occlusions = findOcclusions(track_ids, data[frame])
        for (idA, idB) in occlusions:
            print "La persona ", idB, "pasa por detras de", idA

        crossDetector.addOcclusions(occlusions, frame)

        if display:
            # draw rectangles
            for id in track_ids:
                bbox, found = parseData(data[frame][id])
                if found:
                    cv2.rectangle(image, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), colors[id], 1)
                    cv2.putText(image, str(id), (bbox.xmin, bbox.ymin + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[id], 1)
            cv2.putText(image, str(frame), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (1, 1, 1), 1)

            # draw occlusions
            for (idA, idB) in occlusions:
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

            # display image
            cv2.imshow(dataset, image)

            if cv2.waitKey(500 if len(occlusions) else 1) & 0xff == 27:
                break

        # next
        success, image = vidcap.read()
        frame += 1

    # print detections
    crossDetector.endVideo(frame)
    for occlusion in crossDetector.getCrosses():
        print "Found Occlusion", occlusion


if __name__ == '__main__':
    evalOne('Laboratory/6p-c0', True)
