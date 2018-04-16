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

visor = Visor()
while _somehing_ and visor.hasFinished():
    visor.imshow(_frame_)
visor.finish() # pass False to keep the visor until the user closes it, otherwise it will be automatically closed

"""
import time
from threading import Thread

import cv2

from groundTruthParser import getVideo, getDatasets


class Visor:
    frames = []
    nFrames = 0
    dispFrame = 0
    finished = False
    auto = True

    LABEL = "custom Visor"
    FPS = 60

    def __init__(self, label="custom Visor", fps=60):
        self.LABEL = label
        self.FPS = fps
        t = Thread(target=self.main)
        t.daemon = True
        t.start()

    def imshow(self, frame):
        self.frames.append(frame)
        self.nFrames += 1
        print "added frame", self.nFrames

    def hasFinished(self):
        return self.finished

    def finish(self, forceFinish=True):
        if forceFinish:
            self.finished = True
        while not self.finished:
            time.sleep(1)

    def main(self):
        self.dispFrame = 0
        last_frame = -1

        while self.nFrames == 0:
            time.sleep(1)

        while not self.finished:
            local_nFrames = self.nFrames
            print "displaying frame", self.dispFrame, "/", local_nFrames, "AUTO" if self.auto else "MANUAL"

            if self.auto:
                if last_frame != local_nFrames - 1:
                    cv2.imshow(self.LABEL, self.frames[local_nFrames - 1])
                    last_frame = local_nFrames - 1
                self.dispFrame = local_nFrames
            else:
                if last_frame != self.dispFrame:
                    cv2.imshow(self.LABEL, self.frames[self.dispFrame])
                    last_frame = self.dispFrame

            k = cv2.waitKey(1000 / self.FPS) & 0xff

            if k == 27:  # ESC
                break
            elif k == 83 or k == 100:  # right, d
                self.dispFrame += 1
            elif k == 81 or k == 97:  # left, a
                self.dispFrame -= 1
            elif k == 82 or k == 119:  # up, w
                self.dispFrame += 10
            elif k == 84 or k == 115:  # down, s
                self.dispFrame -= 10
            elif k == 80:  # start
                self.dispFrame = 0
            elif k == 87:  # end
                self.dispFrame = local_nFrames
            elif 49 <= k <= 57:  # 1-9 keys
                self.dispFrame = int(local_nFrames * (k - 48) / 10.)
            elif k != 255:  # other
                print "pressed", k
            self.dispFrame = max(0, min(self.dispFrame, local_nFrames))
            self.auto = self.dispFrame == local_nFrames

        cv2.destroyWindow(self.LABEL)
        self.finished = True


if __name__ == "__main__":
    visor = Visor()

    video = getVideo(getDatasets()[0])
    ok, image = video.read()
    while ok and not visor.hasFinished():
        ok, frame = video.read()

        visor.imshow(frame)
        time.sleep(0.1)

    visor.finish()
