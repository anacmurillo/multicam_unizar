from epfl_scripts.groundTruthParser import getSuperDetector


class CachedDetectron:
    def __init__(self):
        self.detector = {}

    def _loadDataset(self, dataset):
        self.detector[dataset] = {}
        _data = getSuperDetector(dataset)

        for iter_frames in _data.iterkeys():
            self.detector[dataset][iter_frames] = []
            for xmin, ymin, xmax, ymax in _data[iter_frames]:
                self.detector[dataset][iter_frames].append(tuple((xmin, ymin, xmax, ymax)))

    def evaluateImage(self, _, label):
        dataset, frame = label.split(" - ")

        if dataset not in self.detector:
            self._loadDataset(dataset)

        return self.detector[dataset][int(frame)]
