"""
Generates a video showing both the dataset and the result of the tracker with colors
"""

import cv2

from colorUtility import getColors
from evaluator import evalFile
from groundTruthParser import parseFile, getVideo

FOLDER = "/videos/"


def generateVideo(filename, tracker):
    """
    Generates the video from the dataset and the tracker
    :param filename: the filename of the video
    :param tracker: the tracker to use
    :return: Nothing (but a file "{filename}_{tracker}.avi" is generated)
    """
    track_ids, data_groundTruth = parseFile(filename)
    n_frames, data_tracker = evalFile(filename, tracker)
    colors_list = getColors(len(track_ids))

    # initialize video output
    video_in = getVideo(filename)
    video_out = cv2.VideoWriter(FOLDER+filename+"/"+tracker+'avi', cv2.VideoWriter_fourcc(*'XVID'), 25.0, (int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    print "Generating..."

    for frame_index in range(n_frames):

        ok, frame = video_in.read()
        if not ok:
            break
        if frame_index % 500 == 0:
            print "frame", frame_index, "/", n_frames

        for id_index, id in enumerate(track_ids):
            xmin, ymin, xmax, ymax, lost, occluded, generated, label = data_groundTruth[frame_index][id]
            bbox_gt = None if lost else [xmin, ymin, xmax, ymax]
            bbox_tr = data_tracker[frame_index].get(id, None)

            drawIfPresent(frame, bbox_gt, "gt", colors_list[id_index], False)
            drawIfPresent(frame, bbox_tr, "tr", colors_list[id_index], True)
            drawLineBetween(frame, bbox_gt, bbox_tr, colors_list[id_index])

        video_out.write(frame)

    video_out.release()
    video_in.release()

    print "...done"


def drawIfPresent(frame, bbox, text, color, tracker):
    """
    Draws the bbox if not None with specified properties
    """
    if bbox is None:
        return

    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(frame, p1, p2, color, (2 if tracker else 1), (4 if tracker else 8))
    cv2.putText(frame, text, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def drawLineBetween(frame, bboxA, bboxB, color):
    """
    Draws a line bewteen both bboxes, or a point, or none
    """
    if bboxA is not None:
        cA = (bboxA[0] / 2 + bboxA[2] / 2, bboxA[1] / 2 + bboxA[3] / 2)

    if bboxB is not None:
        cB = (bboxB[0] / 2 + bboxB[2] / 2, bboxB[1] / 2 + bboxB[3] / 2)

    if bboxA is None and bboxB is not None:
        cv2.drawMarker(frame, cB, color, cv2.MARKER_SQUARE, 2)
    elif bboxA is not None and bboxB is None:
        cv2.drawMarker(frame, cA, color, 1, 2)
    elif bboxA is not None and bboxB is not None:
        cv2.line(frame, cA, cB, color)


if __name__ == '__main__':
    generateVideo("Laboratory/6p-c0", 'BOOSTING')

