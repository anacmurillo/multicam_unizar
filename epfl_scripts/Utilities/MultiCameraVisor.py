import numpy as np

# import cv2
import epfl_scripts.Utilities.cv2Visor as cv2
from epfl_scripts.Utilities.colorUtility import C_WHITE
from epfl_scripts.Utilities.geometry2D_utils import f_multiplyInv, Point2D
from epfl_scripts.Utilities.geometryCam import from3dCylinder, drawVisibilityLines, toInt, prepareBboxForDisplay
from epfl_scripts.groundTruthParser import getCalibrationMatrixFull

cv2.configure(100)


class MultiCameraVisor:
    """
    Allows for easy displaying of multiple cameras and floor
    """
    FLOOR = "__floor__"

    def __init__(self, cameras, cameraName, floorName):
        self.cameras = cameras

        # frames
        self.frames = {}
        self.floorEmpty = None

        self.cameraName = cameraName
        self.floorName = floorName

        # initialize windows
        cv2.namedWindow(cameraName)
        cv2.moveWindow(cameraName, 0, 0)
        cv2.namedWindow(floorName)
        cv2.moveWindow(floorName, 0, 1000)

    def __del__(self):
        for winname in self.frames:
            cv2.destroyWindow(winname)

    def setFrames(self, frames, copy=False):

        # set camera images
        for camera in self.cameras:
            self.frames[camera] = frames[camera] if not copy else frames[camera].copy()

        # set floor image
        if self.floorEmpty is None:
            self.floorEmpty = np.zeros((512, 512, 3), np.uint8)
            # draw visibility lines
            for camera in self.cameras:
                drawVisibilityLines(self.frames[camera], self.floorEmpty, camera)

        self.frames[self.FLOOR] = self.floorEmpty.copy()

    def setCallback(self, callback):
        cv2.setMouseCallback(self.floorName, callback, self.FLOOR)

        def convert(evt, x, y, flg, _):
            inCamera = None
            for camera in self.cameras:
                dx = self.frames[camera].shape[1]
                if x < dx:
                    inCamera = camera
                    break
                x -= dx
            callback(evt, x, y, flg, inCamera)

        cv2.setMouseCallback(self.cameraName, convert)

    def showAll(self):
        # Display cameras
        concatenatedFrames = None
        for camera in self.cameras:
            if concatenatedFrames is None:
                concatenatedFrames = self.frames[camera]
            else:
                concatenatedFrames = np.hstack((concatenatedFrames, self.frames[camera]))
        cv2.imshow(self.cameraName, concatenatedFrames)

        # Display floor
        cv2.imshow(self.floorName, self.frames[self.FLOOR])

    def getKey(self, delay=1):
        return cv2.waitKey(delay)

    def drawBbox(self, bbox, camera, text=None, color=C_WHITE, thickness=1):
        lt, rb = prepareBboxForDisplay(bbox)
        cv2.rectangle(self.frames[camera], lt, rb, color, thickness, 1)
        if text is not None: self.drawText(text, camera, Point2D(lt[0], lt[1] / 2 + rb[1] / 2), color, thickness / 5.)

    def drawText(self, text, camera, point, color=C_WHITE, size=1.):
        cv2.putText(self.frames[camera], text, toInt(point.getAsXY()), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1)

    def drawCylinder(self, cylinder, text=None, color=C_WHITE, thickness=1):
        # draw bboxes
        for camera in self.cameras:
            self.drawBbox(from3dCylinder(camera, cylinder), camera, text, color, thickness)

        # draw circle
        px, py = toInt(cylinder.getCenter().getAsXY())
        point = toInt((px, py))
        cv2.circle(self.frames[self.FLOOR], point, int(cylinder.width), color, thickness=thickness, lineType=8)

        # draw height
        height = toInt((px, py - cylinder.height * 25))
        cv2.line(self.frames[self.FLOOR], point, height, color, thickness, 1)

    def drawFloorPoint(self, point, color=C_WHITE, thickness=1, drawHeadPoint=False):
        # point in cameras
        for camera in self.cameras:

            groundM, (headT, headP), headH, distMul = getCalibrationMatrixFull(camera)

            # bottom point
            px, py = f_multiplyInv(groundM, point).getAsXY()
            cv2.drawMarker(self.frames[camera], (int(px), int(py)), color, 1, 1, thickness)

            # top point
            if drawHeadPoint:
                if headT == 'm':
                    ppx, ppy = f_multiplyInv(headP, point).getAsXY()
                elif headT == 'h':
                    ppx, ppy = px, headP
                else:
                    raise AttributeError("Unknown calibration parameter: " + headT)
                cv2.drawMarker(self.frames[camera], (int(ppx), int(ppy)), color, 1, 1, thickness)

        # point in floor
        pointfloor = toInt(point.getAsXY())
        cv2.drawMarker(self.frames[self.FLOOR], pointfloor, color, 1, 1, thickness)

    def drawLine(self, point1, point2, camera, color=C_WHITE, thickness=1):
        cv2.line(self.frames[camera], toInt(point1.getAsXY()), toInt(point2.getAsXY()), color, 1, thickness)

    def joinBboxes(self, bbox1, bbox2, camera, color=C_WHITE, thickness=1):
        for dx, dy in [(1, 1), (1, -1), (-1, -1), (-1, 1)]:
            self.drawLine(bbox1.getCenter(dx, dy), bbox2.getCenter(dx, dy), camera, color, thickness)


class NoVisor:
    """
    Does nothing, use as a 'no display' Visor
    """

    def __init__(self, cameras, cameraName, floorName):
        pass

    def __del__(self):
        pass

    def setFrames(self, frames, copy=False):
        pass

    def setCallback(self, callback):
        pass

    def showAll(self):
        pass

    def waitKey(self, delay=0):
        return -1

    def drawBbox(self, bbox, camera, color=C_WHITE, thickness=1):
        pass

    def drawCylinder(self, cylinder, color=C_WHITE, thickness=1):
        pass

    def drawFloorPoint(self, point, color=C_WHITE, thickness=1, drawHeadPoint=False):
        pass

    def drawLine(self, point1, point2, camera, color=C_WHITE, thickness=1):
        pass

    def joinBboxes(self, bbox1, bbox2, camera, color=C_WHITE, thickness=1):
        pass
