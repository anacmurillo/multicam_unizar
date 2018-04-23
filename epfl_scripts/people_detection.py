import cv2

from epfl_scripts.Utilities.customVisor import Visor
from epfl_scripts.Utilities.groundTruthParser import getVideo


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness=1):
    for x, y, w, h in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness)


def showDetectorResults(dataset):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cap = getVideo(dataset)
    ok = True
    visor = Visor('feed')
    while ok and not visor.hasFinished():
        ok, frame = cap.read()
        found, w = hog.detectMultiScale(frame, winStride=(4, 4), padding=(32, 32), scale=1.05)
        draw_detections(frame, found)
        visor.imshow(frame)
    visor.finish()


if __name__ == '__main__':
    showDetectorResults("Laboratory/6p-c0")
