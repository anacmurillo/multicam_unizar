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

cv2 = cv2Visor()

# rest of the code. The object will act as the cv2 package, no other changes are required.

"""

from epfl_scripts.Utilities.groundTruthParser import getVideo, getDatasets


class cv2Visor():
    # CONF
    MAXSAVED = 25
    LABEL = "custom Visor"

    frames = []
    local_cv2 = None

    def __init__(self):
        import cv2 as local_cv2
        self.local_cv2 = local_cv2

    def configure(self, max_saved):
        self.MAXSAVED = max_saved

    # Override
    def imshow(self, winname, mat):
        self.local_cv2.imshow(winname, mat)
        self.LABEL = winname
        if len(self.frames) >= self.MAXSAVED:
            self.frames.pop(0)
        self.frames.append(mat)

    #Override
    def waitKey(self, delay):

        repeat = False
        index = len(self.frames)

        while True:

            k = self.local_cv2.waitKey(delay if not repeat else 0) & 0xff

            if k == 27:  # ESC
                break
            elif k == 83 or k == 100:  # right, d
                index += 1
            elif k == 81 or k == 97:  # left, a
                index -= 1
            elif k == 82 or k == 119:  # up, w
                index += 10
            elif k == 84 or k == 115:  # down, s
                index -= 10
            elif k == 80:  # start
                index = 0
            elif k == 87:  # end
                index = len(self.frames) - 1
            elif 49 <= k <= 57:  # 1-9 keys
                index = int(len(self.frames) * (k - 48) / 10.)
            elif k != 255:  # other
                print "pressed", k

            if index < len(self.frames) or repeat:
                repeat = True
                index = max(0, min(index, len(self.frames) - 1))
                self.local_cv2.imshow(self.LABEL, self.frames[index])
            else:
                break

        return k if not repeat else 255

    #Redirect everything else
    def __getattr__(self, name):
        return getattr(self.local_cv2, name)

if __name__ == "__main__":
    cv2 = cv2Visor()

    video = getVideo(getDatasets()[0])
    ok, image = video.read()
    while ok:
        ok, frame = video.read()

        cv2.imshow("TEST", frame)
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif k != 255:
            print k
    cv2.destroyWindow("TEST")
