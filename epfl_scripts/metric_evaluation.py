import math
import matplotlib.colors as pltcolors
import matplotlib.patches as pltpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as pltticker

from evaluator import evalFile
from groundTruthParser import parseFile

threshold_range = 100


def evaluate(filename, tracker):
    track_ids, data_groundTruth = parseFile(filename)

    data_tracker, n_frames = evalFile(filename, False, tracker)

    motp_total, motp_ids = motp(track_ids, n_frames, data_groundTruth, data_tracker)
    print "MOTP: total =", motp_total
    for id in track_ids:
        print "    ", id, "=", motp_ids[id]

    mota_total, mota_ids = mota(track_ids, n_frames, data_groundTruth, data_tracker)
    print "MOTA: total =", mota_total
    for id in track_ids:
        print "    ", id, "=", mota_ids[id]

    # frame legend
    colors = ['black', 'purple', 'blue', 'red', 'green']
    labels = ['Not present', 'Tracker found ghost', 'Tracker didn\'t found', 'Missed', 'Found']
    colormap = pltcolors.LinearSegmentedColormap.from_list('name', colors)
    binaries = getTrackType(track_ids, n_frames, data_groundTruth, data_tracker)
    minorTicks = []
    mayorTicks = []
    for index, id in enumerate(track_ids):
        x = range(len(binaries[id]))
        y = []
        for y_index in x:
            val = binaries[id][y_index]
            val = val if val >= 0 else 1 if val == -3 else 0
            y.append(index - 0.3 + val * 0.6)
        minorTicks.extend([index - 0.3, index + 0.3])
        mayorTicks.append(index)
        plt.scatter(x, y, s=25, c=binaries[id], marker='|', edgecolors='none', cmap=colormap)
    legend = []
    for i in range(len(labels)):
        legend.append(pltpatches.Rectangle((0, 0), 1, 2, fc=colors[i]))
    plt.legend(legend, labels, bbox_to_anchor=(0.5, 1), loc='upper center', ncol=3, fontsize=10)
    plt.xlim([0, n_frames])
    plt.ylim([-0.5, len(track_ids) + 0.5])
    plt.title('Detection - ' + filename + ' - ' + tracker)
    plt.xlabel('frames')
    plt.ylabel('persons')
    plt.yticks(mayorTicks)
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
    # plt.title('precision-recall - ' + filename + ' - ' + tracker)
    # plt.show()


def precision_recall(id, frames, groundtruth, tracker, threshold):
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
            if getSimilarity(bbox_gt, bbox_tr) >= threshold:
                true_positive += 1
            else:
                false_positive += 1  # false_positive: decidido (el tracker ha encontrado algo, pero esta mal)
                false_negative += 1

    # print true_negative, false_positive, false_negative, true_positive

    return true_positive / (true_positive + false_positive), true_positive / (true_positive + false_negative)


def mota(ids, frames, groundtruth, tracker):
    persons = {}
    total = [0., 0., 0., 0.]

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
                if getSimilarity(bbox_gt, bbox_tr) >= 0.5:  # magic number, consider change with distance
                    pass
                else:
                    mme += 1
        persons[id] = 1. - (mt + fpt + mme) / gt
        for i in range(4):
            total[i] += [mt, fpt, mme, gt][i]

    return 1. - (total[0] + total[1] + total[2]) / total[3], persons


# using different evaluation
def motp(ids, frames, groundtruth, tracker):
    persons = {}
    total = [0., 0.]

    for id in ids:
        distance = 0.
        matches = 0.
        for frame in range(frames):
            xmin, ymin, xmax, ymax, lost, occluded, generated, label = groundtruth[frame][id]
            bbox_gt = None if lost else [xmin, ymin, xmax, ymax]
            bbox_tr = tracker[frame].get(id, None)

            if bbox_gt is not None and bbox_tr is not None:
                distance += f_distance(bbox_gt, bbox_tr)  # getSimilarity(bbox_gt, bbox_tr)
                matches += 1
        persons[id] = distance / matches
        for i in range(2):
            total[i] += [distance, matches][i]

    return total[0] / total[1], persons


def getTrackType(ids, frames, groundtruth, tracker):
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
                binary.append(getSimilarity(bbox_gt, bbox_tr))
        binaries[id] = binary

    return binaries


def getSimilarity(boxA, boxB):
    intersection = f_area([max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])])

    union = f_area(boxA) + f_area(boxB) - intersection

    return intersection / union


def f_area(r):
    return (r[2] - r[0]) * 1. * (r[3] - r[1]) if r[2] > r[0] and r[3] > r[1] else 0.;


def f_distance(boxA, boxB):
    return f_euclidian(f_center(boxA), f_center(boxB))


def f_center(box):
    return (box[2] + box[0]) / 2, (box[3] + box[1]) / 2


def f_euclidian(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


if __name__ == '__main__':
    evaluate("Laboratory/6p-c0", 'BOOSTING')
    evaluate("Laboratory/6p-c0", 'KCF')
    # evaluate("Basketball/match5-c2")

    # for filename in getFilenames():
    #    print filename
    #    evaluate(filename)
