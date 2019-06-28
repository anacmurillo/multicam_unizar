from __future__ import print_function

import os
import sys

import epfl_scripts.sergio.Functions_DatasetLaboratory as fdl
from epfl_scripts.Utilities import KEY
from epfl_scripts.Utilities.MultiCameraVisor import MultiCameraVisor, NoVisor
from epfl_scripts.Utilities.cache import cache_function
from epfl_scripts.Utilities.colorUtility import getColors, C_GREY, C_WHITE, C_RED, blendColors, C_BLACK
from epfl_scripts.Utilities.geometry2D_utils import f_iou, Bbox, Point2D, f_euclidian
from epfl_scripts.Utilities.geometry3D_utils import f_averageCylinders
from epfl_scripts.Utilities.geometryCam import f_similarCylinders, to3dCylinder, cutImage, cropBbox
from epfl_scripts.groundTruthParser import getVideo, getGroupedDatasets
from epfl_scripts.trackers.cylinderTracker import CylinderTracker

### Imports ###
try:
    if "OFFLINE" in os.environ:
        raise Exception("Force offline detectron")
    from detectron_wrapper import Detectron
except Exception as e:
    print("Detectron not available, using cached one. Full exception below:")
    print(e)
    from cachedDetectron import CachedDetectron as Detectron

import cv2

"""
Implementation of the algorithm. Main file
"""

### Variables ###

WIN_NAME = "Tracking"
WINF_NAME = WIN_NAME + "_overview"

FRAMES_LOST = 25  # maximum number of frames until the detection is removed

FRAMES_PERSON = 15  # after this number of frames with at least one camera tracking a target (id<0), it is assigned as person (id>=0)

MAXTRACKERS = 6  # force N trackers at most. <0 to disable

MAX_DIST = 35


### Functions

class Prediction:
    """
    Represents a tracker of a person (or not)
    """
    NEXTPERSONUID = 0
    NEXTUID = -1
    colors = getColors(12)

    def __init__(self, tracker):
        self.tracker = tracker
        self.framesLost = 0
        self.trackerFound = False
        self.detectorFound = False

        self.person = False
        self.uniqueID = Prediction.NEXTUID
        Prediction.NEXTUID -= 1

        self.redefineCylinder = None

    def redefine(self, cylinder):
        """
        Marks the cylinder for the new measure step
        :param cylinder:
        :return:
        """
        self.redefineCylinder = cylinder

    def updateCylinder(self, images):
        """
        Runs the update step of the tracker
        """

        self.trackerFound, _ = self.tracker.update(images, self.redefineCylinder)

    def setPersonIfRequired(self):
        """
        Checks if this tracker meets the condition to be following a person and updates it (does nothing if it already is)
        """
        if self.person: return

        if self.framesLost < -FRAMES_PERSON:
            self.person = True
            self.uniqueID = Prediction.NEXTPERSONUID
            Prediction.NEXTPERSONUID += 1

    def updateLost(self, weight):
        """
        Checks if this tracker meets the condition to be lost
        :param weight: if 1 normal, 2 means twice as fast (for lost), etc
        """
        if self.trackerFound and self.detectorFound:
            framesLost = min(0, self.framesLost - 1)  # if negative, found
        else:
            framesLost = max(0, self.framesLost + 1)  # if positive, lost

        if framesLost < FRAMES_LOST * weight:
            self.framesLost = framesLost
            return False
        else:
            # lost
            return True

    def isWorstThan(self, other):
        """
        returns true iff we are worst than other
        more lost is worst
        smaller id is worst
        we are worst iff we are similar, and we have lost it more
        """

        return f_similarCylinders(self.tracker.getCylinder(), other.tracker.getCylinder()) \
               and (self.framesLost > other.framesLost
                    or (self.framesLost == other.framesLost and self.uniqueID < other.uniqueID)
                    )

    def getColor(self):
        return self.colors[self.uniqueID % len(self.colors)] if self.person else C_WHITE


def onlyUnique(detections, threshold=0):
    """
    Filters the detections and returns only those whose iou with the others are less_or_equal than the threshold
    :param detections:
    :param threshold:
    :return:
    """
    uniques = []

    for detection, _mask in detections:
        valid = True
        for detection2, _mask2 in detections:
            if detection == detection2: continue

            if f_iou(detection, detection2) > threshold:
                valid = False
                break
        if valid:
            uniques.append((detection, _mask))

    return uniques


### Main ###

def assignDetectorToPrediction(cameras, detector, predictions, frames, visor):
    """
    Main function for assignment.
    Using image features
    Returns detections not used
    """
    # persons = [prediction for prediction in predictions if prediction.person]
    # targets = [prediction for prediction in predictions if not prediction.person]

    detector = {camera: onlyUnique(detector[camera]) for camera in cameras}

    detector_unused = {}
    detector_BBM = {}
    for camera in cameras:
        # copy
        detector_unused[camera] = detector[camera][:]

        # create detection array
        detector_BBM[camera] = []
        for detection, mask in detector[camera]:
            image = cutImage(frames[camera], detection)
            detector_BBM[camera].append((image, mask, detection))

    for prediction in predictions:
        bboxes = prediction.tracker.getBboxes()

        # get all bboxes that correspond to this person
        allCylinders = []
        allWeights = []

        for camera in cameras:
            # for each camera

            # compute BB and mask
            BB_p = cutImage(frames[camera], bboxes[camera])
            cylinder_p = to3dCylinder(camera, bboxes[camera])
            if BB_p is None: continue

            # find the best bbox
            best_detection = None
            for BB_d, mask_d, detection in detector_BBM[camera]:

                mask_p = cv2.resize(mask_d, BB_p.shape[1::-1], interpolation=cv2.INTER_NEAREST)

                score, _ = fdl.IstheSamePerson(BB_d, mask_d, BB_p, mask_p, False, fdl)

                if prediction.person:
                    visor.drawText(str(score), camera, detection.getCenter(), color=prediction.getColor())

                if score == 7 and f_iou(detection, bboxes[camera]) > 0.2:
                    # Use only if best score and not very different (iuo)
                    if best_detection is None:
                        best_detection = detection

                        # remove from unused
                        if best_detection in detector_unused[camera]: detector_unused[camera].remove(best_detection)
                    elif best_detection is not False:
                        best_detection = False

            # if good enough, use
            if best_detection is not None and best_detection is not False:
                # use the detection
                cilynder = to3dCylinder(camera, best_detection)
                dist = f_euclidian(cilynder.getCenter(), cylinder_p.getCenter())

                if dist < MAX_DIST:
                    # only near ones
                    allCylinders.append(cilynder)
                    allWeights.append(1. / (1+dist))

                    # show join
                    if prediction.person:
                        visor.joinBboxes(bboxes[camera], best_detection, camera, color=blendColors(prediction.getColor(), C_BLACK), thickness=1)
                else:
                    print (dist)

        if len(allCylinders) > 0:
            # if cylinders to average, average
            cylinder = f_averageCylinders(allCylinders, allWeights)

            prediction.redefine(cylinder)

            prediction.detectorFound = True
        else:
            # no cylinders found, lost
            prediction.detectorFound = False

    return detector_unused


def estimateFromPredictions(predictions, detector, cameras, frames, visor):
    """
    Main function for processing
    """
    # predictions = [prediction class]
    # detector[camera] = [ (bbox), ... ]
    # for camera in cameras: ...
    # frames[camera] = frame

    # phase 1: update bbox with detection (if available)
    if detector is not None:
        # assign detector instances to predictions
        detector = assignDetectorToPrediction(cameras, detector, predictions, frames, visor)

    for prediction in predictions[:]:
        # phase 2.1: remove if lost enough times
        if prediction.updateLost(1 if prediction.person else 2):
            predictions.remove(prediction)

    # phase 3: assign unused detections to new targets
    if detector is not None:
        # for each bbox
        for camera in cameras:
            for bbox, _ in detector[camera]:
                group = {camera: bbox}
                cylinder = to3dCylinder(camera, bbox)
                # find other similar in other cameras
                for camera2 in cameras:
                    if camera == camera2: continue
                    for bbox2, _mask in detector[camera2][:]:
                        if f_similarCylinders(cylinder, to3dCylinder(camera2, bbox2)):
                            group[camera2] = bbox2
                            detector[camera2].remove((bbox2,_mask))

                # create new tracker
                tracker = CylinderTracker(cameras)
                tracker.init(frames, group)
                if MAXTRACKERS >= 0 and len(predictions) >= MAXTRACKERS: break  # debug to keep N trackers at most
                predictions.append(Prediction(tracker))

                for camera in cameras:
                    if camera in group:
                        visor.drawBbox(group[camera], camera, color=C_RED, thickness=2)

    # phase 4: remove duplicated trackers
    uniquePredictions = []
    for prediction in predictions:
        keep = True
        for prediction2 in predictions:
            if prediction.isWorstThan(prediction2):
                keep = False
                # deleted
                if prediction.person:
                    print("deleted person because it is worst than ", prediction2.uniqueID)
        if keep:
            uniquePredictions.append(prediction)

    predictions = uniquePredictions

    # phase 5: set target as person if tracked continuously
    for prediction in predictions:
        prediction.setPersonIfRequired()

    return predictions


@cache_function("evalMultiTracker_{0}_{DETECTOR_FIRED}", lambda _gd, display, DETECTOR_FIRED: cache_function.TYPE_DISABLE if display else cache_function.TYPE_NORMAL, 10)
def evalMultiTracker(groupDataset, display=True, DETECTOR_FIRED=5):
    """
    Main
    groupDataset: lista de datasets
    tracker_type: no utilizado
    display: mostrar por pantalla o no
    detector_fired, cada cuantos frames ejecutar el detector
    """

    # get detector
    detector = Detectron()

    # Visor
    visor = MultiCameraVisor(groupDataset, WIN_NAME, WINF_NAME) if display else NoVisor()

    # initialize videos
    videos = {}
    frames = {}
    nframes = 0
    for dataset in groupDataset:
        video = getVideo(dataset)
        videos[dataset] = video

        # Exit if video not opened.
        if not video.isOpened():
            print("Could not open video for dataset", dataset)
            sys.exit()

        nframes = max(nframes, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))

        # Read first frame.
        ok, frame = video.read()
        frames[dataset] = frame
        if not ok:
            print("Cannot read video file for dataset", dataset)
            sys.exit()

    # initialize detection set
    data_detected = {}
    for dataset in groupDataset:
        data_detected[dataset] = {}

    # initialize predictions
    predictions = []

    # loop
    frame_index = 0
    allOk = True
    while allOk:

        visor.setFrames(frames)

        # parse trackers
        for prediction in predictions:
            prediction.updateCylinder(frames)

        # run detector
        if frame_index % DETECTOR_FIRED == 0:
            detector_results = {}
            for dataset in groupDataset:
                data = detector.evaluateImage(frames[dataset], str(dataset) + " - " + str(frame_index))
                detector_results[dataset] = [(Bbox.XmYmXMYM(result[0], result[1], result[2], result[3]), mask) for result, mask in data]

                # show detections
                for bbox, _ in detector_results[dataset]:
                    visor.drawBbox(bbox, dataset, color=C_GREY)
        else:
            detector_results = None

        # merge all predictions -> estimations
        estimations = estimateFromPredictions(predictions, detector_results, groupDataset, frames, visor)

        # initialize output
        for dataset in groupDataset:
            data_detected[dataset][frame_index] = {}

        # compute detections
        for estimation in estimations:
            bboxes = estimation.tracker.getBboxes()
            for dataset in groupDataset:

                bbox = cropBbox(bboxes[dataset], frames[dataset])
                if bbox is None: continue

                if estimation.person:
                    data_detected[dataset][frame_index][estimation.uniqueID] = bbox.getAsXmYmXMYM()  # xmin, ymin, xmax, ymax

            # show cylinders
            color = estimation.getColor()
            label = "{0}:{1}".format(estimation.uniqueID, estimation.framesLost)
            thickness = 2 if estimation.person else 1
            visor.drawCylinder(estimation.tracker.getCylinder(), text=label, color=color, thickness=thickness)

        # draw frame index
        visor.drawText(str(frame_index), visor.FLOOR, Point2D(0, 512), size=2)

        # show and wait
        visor.showAll()
        if visor.getKey() & 0xff == KEY.ESC:
            break

        if not display:
            # show progress
            # if sys.stdout.isatty():
            sys.stdout.write("\r" + str(frame_index) + "/" + str(nframes) + "     ")
            sys.stdout.flush()

        # read new frames
        for dataset in groupDataset:
            # Read a new frame
            ok, frames[dataset] = videos[dataset].read()
            allOk = allOk and ok

        predictions = estimations

        frame_index += 1

    # clean
    print("")

    for dataset in groupDataset:
        videos[dataset].release()

    return frame_index, range(Prediction.NEXTPERSONUID), data_detected


if __name__ == '__main__':
    # choose dataset
    # dataset = getGroupedDatasets()['Terrace/terrace1']
    # dataset = getGroupedDatasets()['Passageway/passageway1']
    dataset = getGroupedDatasets()['Laboratory/6p']
    # dataset = getGroupedDatasets()['Campus/campus7']

    # v = dataset[0]
    # dataset = [v]

    # choose parameter
    detector_fired = 1

    # run
    print(dataset, detector_fired)
    evalMultiTracker(dataset, display=True, DETECTOR_FIRED=detector_fired)
