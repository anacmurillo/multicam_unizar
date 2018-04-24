"""
Custom visor that allows rewind.
Keys:
right, left (a, d): show +- 1 frame.
up, down (w, s): show +- 10 frames
start: show frame 0
end: show last frame (and activate auto-show)
1-9 keys: show the 10-90% frame
ESC: stop

Normal usage:

import cv2Visor as cv2 # instead of 'import cv2'

# rest of the code. The object will act as the cv2 package, no other changes are required.

"""
import cv2

# CONF
__MAXSAVED = 50
__LABEL = "custom Visor"

__frames = []


def configure(max_saved):
    global __MAXSAVED
    __MAXSAVED = max_saved


# Override
def imshow(winname, mat):
    cv2.imshow(winname, mat)
    global __LABEL
    __LABEL = winname
    if len(__frames) >= __MAXSAVED:
        __frames.pop(0)
    __frames.append(mat)


# Override
def waitKey(delay):
    _repeat = False
    _index = len(__frames)

    while True:

        k = cv2.waitKey(delay if not _repeat else 0) & 0xff

        if k == 27:  # ESC
            break
        elif k == 83 or k == 100:  # right, d
            _index += 1
        elif k == 81 or k == 97:  # left, a
            _index -= 1
        elif k == 82 or k == 119:  # up, w
            _index += 10
        elif k == 84 or k == 115:  # down, s
            _index -= 10
        elif k == 80:  # start
            _index = 0
        elif k == 87:  # end
            _index = len(__frames) - 1
        elif 49 <= k <= 57:  # 1-9 keys
            _index = int(len(__frames) * (k - 48) / 10.)
        elif k != 255:  # other
            print "pressed", k

        if _index < len(__frames) or _repeat:
            _repeat = True
            _index = sorted((0, _index, len(__frames) - 1))[1]  # instead of min(a,max(b,c)) because cv2 redefines min and max
            cv2.imshow(__LABEL, __frames[_index])
        else:
            break

    return k if not _repeat else 255


if __name__ == "__main__":
    def test():
        import sys
        cv2 = sys.modules[__name__]

        import numpy as np

        blank_image = np.zeros((100, 100, 3), np.uint8)

        for i in range(100):

            frame = blank_image.copy()

            cv2.putText(frame, str(i), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("TEST", frame)
            k = cv2.waitKey(100) & 0xff
            if k == 27:
                break
            elif k != 255:
                print k
        cv2.destroyWindow("TEST")
    test()
