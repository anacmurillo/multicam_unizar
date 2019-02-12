"""
Custom visor that allows pausing and rewind up to 50 frames (configurable).
To pause it press the SPACE key (configurable)

Keys:
right, left (a, d): show +- 1 frame.
up, down (w, s): show +- 10 frames
start: show first saved frame
end: show last frame
space: computes the next frame and pauses again (same as ESC+SPACE instantly)
ESC: resume execution


Normal usage:

import cv2Visor as cv2 # instead of 'import cv2'
# rest of the code. The object will act as the cv2 package, no other changes are required.
# optional to configure, example: cv2.configure(100, 13)

"""
import cv2
# noinspection PyUnresolvedReferences
from cv2 import *
# noinspection PyUnresolvedReferences
from cv2 import __version__

# CONF
__MAXSAVED = 50
__PAUSEKEY = 32  # space
__frames = {}
__indexes = {}
__step = False


def configure(max_saved, pause_key):
    """
    Configure the visor properties
    :param max_saved: maximum frames to save (for each window)
    :param pause_key: When pressed, the execution will be paused
    :return:
    """
    global __MAXSAVED
    __MAXSAVED = max_saved

    global __PAUSEKEY
    __PAUSEKEY = pause_key


# Override
def imshow(winname, mat):
    """
    Same as cv2.imshow, but the mat is saved to later use
    :param winname: name of window
    :param mat: frame to show
    :return: None
    """
    cv2.imshow(winname, mat)
    _frames = __frames.setdefault(winname, [])
    if len(_frames) > __MAXSAVED:
        _frames.pop(0)
    _frames.append(mat)
    __indexes[winname] = len(_frames) - 1


# Override
def waitKey(delay):
    """
    Same as cv2.waitKey but pressing the defined key (space by default) pauses the execution and allows rewinding.
    :param delay: delay to wait for keys (can be more if user pauses)
    :return: same as cv2.waitKey (unless user paused and exited, which will return 255)
    """

    global __step
    if __step:
        # nextStep, stop again once
        __step = False
    else:
        _k = cv2.waitKey(delay)
        if (_k & 0xff) != __PAUSEKEY:
            # no pause key, return
            return _k

    # pause
    __pauseDisplay()
    return 255  # no key


def __pauseDisplay():
    """
    Internal pause display, see header
    :return: nothing
    """
    global __step


    while True:

        # change titles
        for _winname in __frames:
            currentFrame = len(__frames[_winname]) - 1 - __indexes[_winname]
            cv2.setWindowTitle(_winname, "{} *PAUSED{}*".format(_winname, " " + str(-currentFrame) if currentFrame != 0 else ""))

        # wait key infinitely
        _k = cv2.waitKey(0) & 0xff
        _index = 0

        # parse key
        if _k == 27:  # ESC
            break
        elif _k == 83 or _k == 100:  # right, d
            _index = 1
        elif _k == 81 or _k == 97:  # left, a
            _index = -1
        elif _k == 82 or _k == 119:  # up, w
            _index = 10
        elif _k == 84 or _k == 115:  # down, s
            _index = -10
        elif _k == 80:  # start
            _index = - __MAXSAVED
        elif _k == 87:  # end
            _index = __MAXSAVED
        elif _k == 32:  # space
            __step = True
            break
        elif _k != 255:  # other
            print "pressed", _k

        # update displays
        for _winname in __frames:
            __indexes[_winname] = sorted((0, __indexes[_winname] + _index, len(__frames[_winname]) - 1))[1]  # instead of min(a,max(b,c)) because cv2 redefines min and max
            cv2.imshow(_winname, __frames[_winname][__indexes[_winname]])

    # restore titles
    for _winname in __frames:
        cv2.setWindowTitle(_winname, _winname)


if __name__ == "__main__":
    def __test():
        import sys
        cv2 = sys.modules[__name__]

        import numpy as np

        blank_image = np.zeros((100, 100, 3), np.uint8)

        for i in range(100):

            frame = blank_image.copy()

            cv2.putText(frame, str(i), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("TEST", frame)

            if i % 5 == 0:
                cv2.imshow("TEST2", frame)

            k = cv2.waitKey(100) & 0xff
            if k == 27:
                break
            elif k != 255:
                print k
        cv2.destroyWindow("TEST")


    __test()
