import numpy as np

# import cv2
import epfl_scripts.Utilities.cv2Visor as cv2
from epfl_scripts.Utilities.colorUtility import C_WHITE, invColor, blendColors
from epfl_scripts.Utilities.geometry2D_utils import f_multiplyInv, Point2D, f_area
from epfl_scripts.Utilities.geometryCam import from3dCylinder, drawVisibilityLines, toInt, prepareBboxForDisplay, cropBbox
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
        """
        Sets the frames as the background, resets all previous modifications
        If copy is True the frames will be copied (otherwise when drawing things they will be drawn to the passed frame)
        """

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
        """
        Sets the callback for mouse events.
        The callback must be a fucntions with the following parameters (evt, x, y, flg, dataset) where
        evt = which event
        x,y = position of the event
        flg = flags of the event
        dataset = dataset where the event occurred
        """
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
        """
        Draws the current saved frames to the screen
        """
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
        """
        smae as cv2.waitKey
        :param delay: time to wait for checking keys (0=foreeeever)
        :return: the pressed key (or -1 if timeout)
        """
        return cv2.waitKey(delay)

    def drawBbox(self, bbox, camera, text=None, color=C_WHITE, thickness=1):
        """
        Draws a bounding box in the specified
        :param bbox: Bounding box to draw
        :param camera: which camera to draw into
        :param text: [optional] text to draw inside the bounding box
        :param color: color of the bounding box
        :param thickness: thickness of the bounding box
        """
        lt, rb = prepareBboxForDisplay(bbox)
        cv2.rectangle(self.frames[camera], lt, rb, color, thickness, 1)
        if text is not None: self.drawText(text, camera, Point2D(lt[0], rb[1]), color, thickness / 5.)

    def drawText(self, text, camera, point, color=C_WHITE, size=1.):
        """
        Draws a text
        :param text: Text to draw
        :param camera: which camera to draw into (use this.FLOOR for floor)
        :param point: Bottom-left position of the text
        :param color: color of the bounding box
        :param size: size of the text
        """
        text = str(text)
        cv2.putText(self.frames[camera], text, toInt(point.getAsXY()), cv2.FONT_HERSHEY_SIMPLEX, size, blendColors(color, invColor(color)), 2)
        cv2.putText(self.frames[camera], text, toInt(point.getAsXY()), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1)

    def drawCylinder(self, cylinder, text=None, color=C_WHITE, thickness=1):
        """
        Draws a cylinder. A circle with a line in the floor, and bounding boxes in the cameras.
        :param cylinder: Cylinder to draw
        :param text: [optional] text to draw inside the bounding boxes (not the circle)
        :param color: color of the circle and the bounding boxes
        :param thickness: thickness of the circle and the bounding boxes
        """
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
        """
        Draws a point in the floor and the cameras.
        :param point: Point to draw (in floor coordinates)
        :param color: color of the points
        :param thickness: thickness of the points
        :param drawHeadPoint: if true, the head point will also be drawed in the cameras (point from the same x,y but from the head plane
        """
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
        """
        Draws a line
        :param point1: One end of the line
        :param point2: The other end of the line
        :param camera: which camera to draw into
        :param color: color of the line
        :param thickness: thickness of the line
        """
        cv2.line(self.frames[camera], toInt(point1.getAsXY()), toInt(point2.getAsXY()), color, 1, thickness)

    def joinBboxes(self, bbox1, bbox2, camera, color=C_WHITE, thickness=1):
        """
        Draws a line between each corner of the two bounding boxes
        :param bbox1: One bounding box
        :param bbox2: The other bounding box
        :param camera: which camera to draw into
        :param color: color of the line
        :param thickness: thickness of the lines
        """
        for dx, dy in [(1, 1), (1, -1), (-1, -1), (-1, 1)]:
            self.drawLine(bbox1.getCenter(dx, dy), bbox2.getCenter(dx, dy), camera, color, thickness)

    def drawImage(self, image, bbox, camera):
        bbox = cropBbox(bbox, self.frames[camera])

        if f_area(bbox) > 0:
            self.frames[camera][bbox.ymin:bbox.ymax + 1, bbox.xmin:bbox.xmax + 1] = image


class NoVisor(object):
    """
    Does nothing, use as a 'no display' Visor
    """

    FLOOR = None  # unfortunately static things can't be placed in a super function

    @staticmethod
    def __global(attr):
        # functions that return something, otherwise return None
        returns = {'getKey': -1}

        return lambda *args, **kwargs: returns.get(attr)

    def __getattr__(self, attr):
        try:
            # existent parameter (something from python)
            return super(NoVisor, self).__getattr__(attr)
        except AttributeError:
            # something nonexistent (something from us)
            return self.__global(attr)
