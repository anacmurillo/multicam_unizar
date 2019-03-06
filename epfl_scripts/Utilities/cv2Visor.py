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
__step = False


def configure(max_saved=None, pause_key=None):
    """
    Configure the visor properties
    :param max_saved: maximum frames to save (for each window)
    :param pause_key: When pressed, the execution will be paused
    :return:
    """
    if max_saved is not None:
        global __MAXSAVED
        __MAXSAVED = max_saved

    if pause_key is not None:
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
    r = cv2.imshow(winname, mat)

    frames = __frames.setdefault(winname, [])  # set if nonexistent, and get
    if len(frames) > __MAXSAVED:
        frames.pop()
    frames.insert(0, mat)
    return r


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
        k = cv2.waitKey(delay)
        if (k & 0xff) != __PAUSEKEY:
            # no pause key, return
            return k

    # pause
    __pauseDisplay()
    return 255  # no key


def destroyWindow(winname):
    """
    Same as cv2.destroyWindow, removes saved images from the specified winname too
    """
    r = cv2.destroyWindow(winname)

    if winname in __frames:
        del __frames[winname]

    return r


def destroyAllWindows():
    """
    Same as cv2.destroyAllWindows, removes all saved images too
    """
    r = cv2.destroyAllWindows()

    __frames.clear()

    return r


def __pauseDisplay():
    """
    Internal pause display, see header
    :return: nothing
    """
    global __step

    # init
    indexes = {}  # 0 -> latest, len() -> oldest
    for winname in __frames:
        indexes[winname] = 0

    while True:
        # change titles
        for winname in __frames:
            if indexes[winname] == 0:
                cv2.setWindowTitle(winname, "{} *PAUSED*".format(winname))
            else:
                cv2.setWindowTitle(winname, "{} *REWIND (-{}/{})*".format(winname, indexes[winname], len(__frames[winname]) - 1))

        # wait for key infinitely
        k = cv2.waitKey(0) & 0xff
        index = 0

        # parse key
        if k == 27:  # ESC
            break
        elif k == 83 or k == 100:  # right, d
            index = 1
        elif k == 81 or k == 97:  # left, a
            index = -1
        elif k == 82 or k == 119:  # up, w
            index = 10
        elif k == 84 or k == 115:  # down, s
            index = -10
        elif k == 80:  # start
            index = - __MAXSAVED
        elif k == 87:  # end
            index = __MAXSAVED
        elif k == 32:  # space
            __step = True
            break
        elif k != 255:  # other
            print "pressed", k

        # update displays
        for winname in __frames:
            indexes[winname] = sorted((0, indexes[winname] - index, len(__frames[winname]) - 1))[1]  # instead of min(a,max(b,c)) because cv2 redefines min and max
            cv2.imshow(winname, __frames[winname][indexes[winname]])

    # fast forward to latest image at double speed
    while sorted(indexes.values())[-1]:
        for winname in __frames:
            if indexes[winname] >= 0:
                cv2.setWindowTitle(winname, "{} *FASTFORWARD*".format(winname))
                cv2.imshow(winname, __frames[winname][indexes[winname]])
                indexes[winname] -= 2  # for faster fastforward

        if cv2.waitKey(1) & 0xff == 27:
            # ESC, stop fastforward
            break

    # restore titles and latest images
    for winname in __frames:
        cv2.setWindowTitle(winname, winname)
        if indexes[winname] != 0:
            cv2.imshow(winname, __frames[winname][0])


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
