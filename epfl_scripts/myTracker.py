"""
Custom implementation of an 'advanced' tracker
You can redefine bboxes
"""


class MyTracker:
    """
    Custom tracker implementation to use as a cv2 tracker.
    """

    def __init__(self):
        self.lastBbox = None  # (xmin, ymin, width, height)
        self.diff = None

    def init(self, frame, bbox):
        self.lastBbox = bbox
        self.diff = (0, 0, 0, 0)
        return True

    def update(self, frame):
        self.diff = tuple(self.diff[i] * 0.75 for i in range(4))
        bbox = tuple(self.lastBbox[i] + self.diff[i] for i in range(4))
        self.lastBbox = bbox

        intbbox = tuple(int(bbox[i]) for i in range(4))
        return True, intbbox

    def redefine(self, newbbox):
        # diff = new - prev = new - ( last - diff ) = new - last + diff
        self.diff = tuple(newbbox[i] - self.lastBbox[i] + self.diff[i] for i in range(4))
        self.lastBbox = newbbox


########################### internal #################################


if __name__ == '__main__':
    print "TODO"
