"""
Evaluation of metrics
"""
import math
import matplotlib.colors as pltcolors
import matplotlib.patches as pltpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as pltticker

from epfl_scripts.Utilities.cv2Trackers import evaluateTracker
from epfl_scripts.Utilities.groundTruthParser import getGroundTruth

threshold_range = 100


def evaluateMetrics(dataset, tracker):
    """
    Evaluates the specified dataset with the specified tracker
    :param dataset: the dataset to evaluate
    :param tracker: the tracker to evaluate
    """
    track_ids, data_groundTruth = getGroundTruth(dataset)

    n_frames, data_tracker = evaluateTracker(dataset, tracker)

    # MOTP
    print "MOTP:"
    motp_ids = motp(track_ids, n_frames, data_groundTruth, data_tracker)
    motp_average = 0
    for id in track_ids:
        print "    ", id, "=", motp_ids[id]
    print "average =", motp_average / len(track_ids)

    # MOTA
    print "MOTA"
    mota_ids = mota(track_ids, n_frames, data_groundTruth, data_tracker)
    mota_average = 0
    for id in track_ids:
        mota_average += mota_ids[id]
        print "    ", id, "=", mota_ids[id]
    print "average =", mota_average / len(track_ids)

    # frame legend
    colors = ['black', 'purple', 'blue', 'red', 'green']
    labels = ['Not present', 'Tracker found ghost', 'Tracker didn\'t found', 'Totally Missed (IOU=0)', 'Perfect Found (IUO=1)']
    colormap = pltcolors.LinearSegmentedColormap.from_list('name', colors)
    binaries = getTrackType(track_ids, n_frames, data_groundTruth, data_tracker)
    minorTicks = []
    for index, id in enumerate(track_ids):
        x = range(len(binaries[id]))
        y = []
        for y_index in x:
            val = binaries[id][y_index]
            val = val if val >= 0 else 1 if val == -3 else 0
            y.append(index - 0.3 + val * 0.6)
        minorTicks.extend([index - 0.3, index + 0.3])
        plt.scatter(x, y, s=25, c=binaries[id], marker='|', edgecolors='none', cmap=colormap)
    legend = []
    for i in range(len(labels)):
        legend.append(pltpatches.Rectangle((0, 0), 1, 2, fc=colors[i]))
    plt.legend(legend, labels, bbox_to_anchor=(0.5, 1), loc='upper center', ncol=3, fontsize=10)
    plt.xlim([0, n_frames])
    plt.ylim([-0.5, len(track_ids) + 0.5])
    plt.title('Detection - ' + dataset + ' - ' + tracker)
    plt.xlabel('frames')
    plt.ylabel('persons')
    plt.yticks(*zip(*list(enumerate(track_ids))))
    plt.gca().yaxis.set_minor_locator(pltticker.FixedLocator(minorTicks))
    plt.grid(True, which='major', axis='y', linestyle=':')
    plt.grid(True, which='minor', axis='y', linestyle='-')
    plt.show()

    # for id in track_ids:
    #     x = []
    #     y = []
    #     c = []
    #     for threshold in [i * 1. / threshold_range for i in xrange(0, threshold_range + 1)]:
    #         precision, recall = precision_recall(id, n_frames, data_groundTruth, data_tracker, threshold)
    #         y.append(precision)
    #         x.append(recall)
    #         c.append(threshold)
    #
    #     scttr = plt.scatter(x, y, c=c, edgecolors='none')
    # plt.colorbar(scttr)
    # plt.xlabel('recall')
    # plt.xlim([0, 1])
    # plt.ylabel('precision')
    # plt.ylim([0, 1])
    # plt.title('precision-recall - ' + dataset + ' - ' + tracker)
    # plt.show()


def precision_recall(id, frames, groundtruth, tracker, threshold):
    """
    returns precision and recall in general
    """
    true_negative = 0.  # person not in groundtruth and not found by tracker
    false_positive = 0.  # tracker found person not in groundtruth
    false_negative = 0.  # person in groundtruth not found by tracker
    true_positive = 0.  # tracker and groundtruth found the same person

    for frame in range(frames):
        xmin, ymin, xmax, ymax, lost, occluded, generated, label = groundtruth[frame][id]
        bbox_gt = None if lost else [xmin, ymin, xmax, ymax]
        bbox_tr = tracker[frame].get(id, None)

        if bbox_gt is None and bbox_tr is None:
            true_negative += 1
        elif bbox_gt is None and bbox_tr is not None:
            false_positive += 1
        elif bbox_gt is not None and bbox_tr is None:
            false_negative += 1
        else:
            if f_iou(bbox_gt, bbox_tr) >= threshold:
                true_positive += 1
            else:
                false_positive += 1  # false_positive: decidido (el tracker ha encontrado algo, pero esta mal)
                false_negative += 1

    # print true_negative, false_positive, false_negative, true_positive

    return true_positive / (true_positive + false_positive), true_positive / (true_positive + false_negative)


def mota(ids, frames, groundtruth, tracker):
    """
    Returns the mota evaluation of each person
    """
    persons = {}

    for id in ids:
        mt = 0.  # number of misses (persons not found)
        fpt = 0.  # number of false positives (persons found but not in groundtruth)
        mme = 0.  # number of mismatches (persons found but not from this groundtruth)
        gt = 0.  # number of groundtruth available
        for frame in range(frames):
            xmin, ymin, xmax, ymax, lost, occluded, generated, label = groundtruth[frame][id]
            bbox_gt = None if lost else [xmin, ymin, xmax, ymax]
            bbox_tr = tracker[frame].get(id, None)

            if bbox_gt is not None:
                gt += 1

            if bbox_gt is None and bbox_tr is None:
                pass
            elif bbox_gt is None and bbox_tr is not None:
                fpt += 1
            elif bbox_gt is not None and bbox_tr is None:
                mt += 1
            else:
                if f_iou(bbox_gt, bbox_tr) >= 0.5:  # magic number, consider change with distance
                    pass
                else:
                    mme += 1
        persons[id] = 1. - (mt + fpt + mme) / gt

    return persons


# using different evaluation
def motp(ids, frames, groundtruth, tracker):
    """
    Returns the motp evaluation for each person
    """
    persons = {}

    for id in ids:
        distance = 0.
        matches = 0.
        for frame in range(frames):
            xmin, ymin, xmax, ymax, lost, occluded, generated, label = groundtruth[frame][id]
            bbox_gt = None if lost else [xmin, ymin, xmax, ymax]
            bbox_tr = tracker[frame].get(id, None)

            if bbox_gt is not None and bbox_tr is not None:
                distance += f_distance(bbox_gt, bbox_tr)  # f_iou(bbox_gt, bbox_tr)
                matches += 1
        persons[id] = distance / matches

    return persons


def getTrackType(ids, frames, groundtruth, tracker):
    """
    Returns the result of comparing the groundtruth and the tracker for each frame of each person
    :return: a dictionary, keys are ids and values are list for each frame
    Value for a frame of a person:
        -3    -> person not in groundtruth and not in tracker
        -2    -> person not in groundtruth but in tracker
        -1    -> person in groundtruth but not in tracker
        [0,1] -> person in groundtruth and in tracker. Value is the IUO of both boxes
    """
    binaries = {}

    for id in ids:
        binary = []  # type of each frame
        for frame in range(frames):
            xmin, ymin, xmax, ymax, lost, occluded, generated, label = groundtruth[frame][id]
            bbox_gt = None if lost else [xmin, ymin, xmax, ymax]
            bbox_tr = tracker[frame].get(id, None)

            if bbox_gt is None and bbox_tr is None:
                binary.append(-3)
            elif bbox_gt is None and bbox_tr is not None:
                binary.append(-2)
            elif bbox_gt is not None and bbox_tr is None:
                binary.append(-1)
            else:
                binary.append(f_iou(bbox_gt, bbox_tr))
        binaries[id] = binary

    return binaries


######################### utilities #######################
# note: bbox means (xmin, ymin, xmax, ymax)


def f_iou(boxA, boxB):
    """
    IOU (Intersection over Union) of both boxes.
    :return: value between [0.1]. 0 if disjointed bboxes, 1 if equal bboxes
    """
    intersection = f_area([max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])])

    union = f_area(boxA) + f_area(boxB) - intersection

    return intersection / union


def f_area(bbox):
    """
    return area of bbox
    """
    return (bbox[2] - bbox[0]) * 1. * (bbox[3] - bbox[1]) if bbox[2] > bbox[0] and bbox[3] > bbox[1] else 0.


def f_distance(boxA, boxB):
    """
    return eucliedean distance bewteen center of bboxes
    """
    return f_euclidian(f_center(boxA), f_center(boxB))


def f_center(box):
    """
    Returns the center of the bbox
    """
    return (box[2] + box[0]) / 2, (box[3] + box[1]) / 2


def f_euclidian(a, b):
    """
    returns the euclidian distance bewteen the two points
    """
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


if __name__ == '__main__':
    evaluateMetrics("Laboratory/6p-c0", 'MEANSHIFT')
    evaluateMetrics("Laboratory/6p-c0", 'CAMSHIFT')
    # evaluateMetrics("Basketball/match5-c2")

    # for dataset in getDatasets():
    #    print dataset
    #    evaluateMetrics(dataset)
