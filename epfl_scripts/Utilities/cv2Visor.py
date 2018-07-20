"""
Custom visor that allows rewind up to 50 frames (configurable).
Keys:
right, left (a, d): show +- 1 frame.
up, down (w, s): show +- 10 frames
start: show first saved frame
end: show last frame
ESC: stop


Normal usage:

import cv2Visor as cv2 # instead of 'import cv2'
# rest of the code. The object will act as the cv2 package, no other changes are required.

"""
import cv2
# noinspection PyUnresolvedReferences
from cv2 import *
# noinspection PyUnresolvedReferences
from cv2 import __version__

# CONF
__MAXSAVED = 50
__frames = {}
__indexes = {}


def configure(max_saved):
    """
    Configure the maximum frames to save
    :param max_saved: maximum frames to save (for each window)
    :return:
    """
    global __MAXSAVED
    __MAXSAVED = max_saved


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
    if len(_frames) >= __MAXSAVED:
        _frames.pop(0)
    _frames.append(mat)
    __indexes[winname] = len(_frames)


# Override
def waitKey(delay):
    """
    Same as cv2.waitKey but pressing an arrow button (or wasd) or start/end will pause the execution and allow the user to review past frames.
    While in review mode press ESC or other button to exit
    :param delay: delay to wait for keys (can be more if user pauses)
    :return: key pressed (unles user paused and exited, which will return 255)
    """
    _repeat = False

    while True:

        _k = cv2.waitKey(delay if not _repeat else 0) & 0xff
        _index = 0

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
        elif _k != 255:  # other
            print "pressed", _k

        if _index != 0 or _repeat:
            _repeat = True
            for _winname in __frames:
                __indexes[_winname] = sorted((0, __indexes[_winname] + _index, len(__frames[_winname]) - 1))[1]  # instead of min(a,max(b,c)) because cv2 redefines min and max
                cv2.imshow(_winname, __frames[_winname][__indexes[_winname]])
        else:
            break

    return _k if not _repeat else 255


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
