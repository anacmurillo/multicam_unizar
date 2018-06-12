"""
Evaluation of metrics
"""
import math
import matplotlib.colors as pltcolors
import matplotlib.patches as pltpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as pltticker
import numpy as np
import sys
from datetime import datetime

from epfl_scripts.Utilities.colorUtility import getColors, blendColors
from epfl_scripts.Utilities.cv2Trackers import evaluateTracker, getTrackers
from epfl_scripts.Utilities.groundTruthParser import getGroundTruth, getGroupedDatasets
from epfl_scripts.multiCameraTrackerV2 import evalMultiTracker


def evaluateMetrics(dataset, tracker):
    """
    Evaluates the specified dataset with the specified tracker
    :param dataset: the dataset to evaluate
    :param tracker: the tracker to evaluate
    """
    track_ids, data_groundTruth = getGroundTruth(dataset)
    n_frames, data_tracker = evaluateTracker(dataset, tracker)
    evaluateData(track_ids, data_groundTruth, n_frames, track_ids, data_tracker, 'Detection - ' + dataset + ' - ' + tracker)


def evaluateMetricsGroup(groupDataset, tracker, toFile=None, detector=5):
    n_frames, n_ids, data = evalMultiTracker(groupDataset, tracker, False, detector)

    motas = []
    motps = []

    if toFile is not None:
        sys.stdout = open(toFile + ".txt", "w")

        print datetime.now()
        print "n_frames =", n_frames, " n_ids =", n_ids
        print

        # save to file
        with open(toFile + ".data", "w") as the_file:
            the_file.write("dataset frame id xmin ymin xmax ymax\n")
            for dataset in groupDataset:
                for frame in range(n_frames):
                    for id in data[dataset][frame]:
                        xmin, ymin, xmax, ymax = data[dataset][frame][id]
                        the_file.write(" ".join(map(str, [dataset, str(frame), id, xmin, ymin, xmax, ymax, "\n"])))

    for dataset in groupDataset:
        gt_ids, data_groundTruth = getGroundTruth(dataset)
        mota_ids, motp_ids = evaluateData(gt_ids, data_groundTruth, n_frames, n_ids, data[dataset], 'Detection - ' + dataset + ' - ' + tracker, False)
        for id in gt_ids:
            if motp_ids[id] >= 0:
                motps.append(motp_ids[id])
            motas.append(mota_ids[id])

    if toFile is not None:
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            fig.savefig(toFile + "_" + str(fig_num) + ".png")
        sys.stdout = sys.__stdout__
    else:
        plt.show()
    plt.close('all')

    return sum(motps) / len(motps), sum(motas) / len(motas)


def evaluateData(gt_ids, data_groundTruth, n_frames, tr_ids, data_tracker, label, block=True):
    print
    print "Metrics of:", label
    if len(tr_ids) != len(gt_ids):
        print "[WARNING] There are", len(gt_ids), "ids on dataset, but", len(tr_ids), "returned by tracker"

    # remove ids changes from tracker
    data_tracker_polished = {}
    for frame in range(n_frames):
        data_tracker_polished[frame] = {}
        for gt_id in gt_ids:
            bbox_gt = getBboxFromGroundtruth(data_groundTruth[frame][gt_id])
            if bbox_gt is None: continue

            bestBbox = None
            bestIou = 0
            for tr_id in tr_ids:
                bbox_tr = data_tracker[frame].get(tr_id, None)
                if bbox_tr is None: continue

                iou = f_iou(bbox_tr, bbox_gt)
                if iou > bestIou:
                    bestBbox = bbox_tr
                    bestIou = iou
            data_tracker_polished[frame][gt_id] = bestBbox

    plt.figure("graph_" + label)
    plt.suptitle(label, fontsize=16)

    # MOTP
    print "MOTP:"
    motp_ids = motp(gt_ids, n_frames, data_groundTruth, data_tracker_polished)
    motp_average = 0.
    for id in gt_ids:
        print "    ", id, "=", motp_ids[id]
        if motp_ids[id] >= 0:
            motp_average += motp_ids[id]
    print "average =", motp_average / len(gt_ids)

    plt.subplot(2, 2, 1)
    plt.barh(range(len(gt_ids)), [motp_ids[id] for id in gt_ids], align='center')
    plt.axvline(x=motp_average / len(gt_ids))
    for i, id in enumerate(gt_ids):
        plt.text(motp_ids[id], i, '%.2f' % motp_ids[id], color='blue', va='center', fontweight='bold')
    plt.xlim([0, 50])
    plt.ylim([-1, len(gt_ids)])
    plt.title("MOTP")
    plt.xlabel('frames')
    plt.ylabel('persons')
    plt.yticks(*zip(*list(enumerate(gt_ids))))

    # MOTA
    print "MOTA"
    mota_ids = mota(gt_ids, n_frames, data_groundTruth, data_tracker_polished)
    mota_average = 0
    for id in gt_ids:
        mota_average += mota_ids[id]
        print "    ", id, "=", mota_ids[id]
    print "average =", mota_average / len(gt_ids)

    plt.subplot(2, 2, 2)
    plt.barh(range(len(gt_ids)), [mota_ids[i] for i in gt_ids], align='center')
    plt.axvline(x=mota_average / len(gt_ids))
    for i, id in enumerate(gt_ids):
        plt.text(mota_ids[id], i, '%.2f' % mota_ids[id], color='blue', va='center', fontweight='bold')
    plt.xlim([-0.5, 1.5])
    plt.ylim([-1, len(gt_ids)])
    plt.title("MOTA")
    plt.xlabel('frames')
    plt.ylabel('persons')
    plt.yticks(*zip(*list(enumerate(gt_ids))))

    # iou_graph
    # plt.figure("iou_graph_" + label)
    plt.subplot(2, 2, 3)
    iou_graph(gt_ids, data_groundTruth, n_frames, data_tracker_polished, label)

    # id_graph
    # plt.figure("id_graph_" + label)
    plt.subplot(2, 2, 4)
    id_graph(gt_ids, data_groundTruth, n_frames, tr_ids, data_tracker, label)

    if block:
        plt.show()

    # for id in gt_ids:
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

    return motp_ids, mota_ids


def iou_graph(gt_ids, data_groundTruth, n_frames, data_tracker, label):
    colors = ['black', 'purple', 'blue', 'red', 'green']
    labels = ['Not present', 'Tracker found ghost', 'Tracker didn\'t found', 'Totally Missed (IOU=0)', 'Perfect Found (IUO=1)']
    colormap = pltcolors.LinearSegmentedColormap.from_list('name', colors)
    normalization = pltcolors.Normalize(vmin=-3, vmax=1)
    binaries = getTrackType(gt_ids, n_frames, data_groundTruth, data_tracker)
    minorTicks = []
    for index, id in enumerate(gt_ids):
        x = range(n_frames)
        y = []
        for frame in x:
            val = binaries[id][frame]
            val = val if val >= 0 else 1 if val == -3 else 0
            y.append(index - 0.3 + val * 0.6)
        minorTicks.extend([index - 0.3, index + 0.3])
        plt.scatter(x, y, s=25, c=binaries[id], marker='|', edgecolors='none', cmap=colormap, norm=normalization)
    legend = []
    for i in range(len(labels)):
        legend.append(pltpatches.Rectangle((0, 0), 1, 2, fc=colors[i]))
    plt.legend(legend, labels, bbox_to_anchor=(0.5, 1), loc='upper center', ncol=3, fontsize=10)
    plt.xlim([0, n_frames])
    plt.ylim([-0.5, len(gt_ids) + 0.5])
    plt.title('best IOU')
    plt.xlabel('frames')
    plt.ylabel('persons')
    plt.yticks(*zip(*list(enumerate(gt_ids))))
    plt.gca().yaxis.set_minor_locator(pltticker.FixedLocator(minorTicks))
    plt.grid(True, which='major', axis='y', linestyle=':')
    plt.grid(True, which='minor', axis='y', linestyle='-')


def id_graph(gt_ids, data_groundTruth, n_frames, tr_ids, data_tracker, label):
    tr_len = len(tr_ids)
    gt_len = len(gt_ids)
    grid = np.zeros([gt_len * tr_len, n_frames, 3], dtype=np.uint8)

    colors = getColors(gt_len)

    for frame in range(n_frames):
        for tr_index, tr_id in enumerate(tr_ids):
            for gt_index, gt_id in enumerate(gt_ids):
                bbox_gt = getBboxFromGroundtruth(data_groundTruth[frame][gt_id])
                bbox_tr = data_tracker[frame].get(tr_id, None)

                if bbox_gt is None and bbox_tr is None:
                    color = [205, 205, 205]
                elif bbox_gt is None and bbox_tr is not None:
                    color = [255, 255, 205]
                elif bbox_gt is not None and bbox_tr is None:
                    color = [205, 205, 255]
                else:
                    color = list(blendColors((255., 255., 255.), colors[gt_index], f_iou(bbox_gt, bbox_tr)))

                grid[tr_index * gt_len + gt_index, frame] = color
    plt.imshow(grid, extent=[0, grid.shape[1], 0, grid.shape[0]], aspect='auto', interpolation='none', origin='lower')
    plt.yticks(*zip(*[(i * gt_len + gt_len / 2., x) for i, x in enumerate(tr_ids)]))
    plt.gca().yaxis.set_minor_locator(pltticker.FixedLocator(range(grid.shape[0])))
    plt.title('all IOU')
    plt.xlabel('frames')
    plt.ylabel('persons (tracker/groundtruth)')
    for tr_index in range(tr_len):
        for gt_index in range(gt_len):
            plt.axhline(y=tr_index * gt_len + gt_index + 1, color='black', linestyle=':' if gt_index != gt_len - 1 else '-')


def precision_recall(id, frames, groundtruth, tracker, threshold):
    """
    returns precision and recall in general
    """
    true_negative = 0.  # person not in groundtruth and not found by tracker
    false_positive = 0.  # tracker found person not in groundtruth
    false_negative = 0.  # person in groundtruth not found by tracker
    true_positive = 0.  # tracker and groundtruth found the same person

    for frame in range(frames):
        bbox_gt = getBboxFromGroundtruth(groundtruth[frame][id])
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
            bbox_gt = getBboxFromGroundtruth(groundtruth[frame][id])
            bbox_tr = tracker[frame].get(id, None)

            if bbox_gt is not None:
                gt += 1.

            if bbox_gt is None and bbox_tr is None:
                pass
            elif bbox_gt is None and bbox_tr is not None:
                fpt += 1.
            elif bbox_gt is not None and bbox_tr is None:
                mt += 1.
            else:
                if f_iou(bbox_gt, bbox_tr) >= 0.5:  # magic number, consider change with distance
                    pass
                else:
                    mme += 1.
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
            bbox_gt = getBboxFromGroundtruth(groundtruth[frame][id])
            bbox_tr = tracker[frame].get(id, None)

            if bbox_gt is not None and bbox_tr is not None:
                distance += f_distance(bbox_gt, bbox_tr)  # f_iou(bbox_gt, bbox_tr)
                matches += 1.
        persons[id] = distance / matches if matches > 0 else -1.

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
            bbox_gt = getBboxFromGroundtruth(groundtruth[frame][id])
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


def getBboxFromGroundtruth(data):
    xmin, ymin, xmax, ymax, lost, occluded, generated, label = data
    return None if lost else [xmin, ymin, xmax, ymax]


######################### utilities #######################
# note: bbox means (xmin, ymin, xmax, ymax)


def f_iou(boxA, boxB):
    """
    IOU (Intersection over Union) of both boxes.
    :return: value in range [0,1]. 0 if disjointed bboxes, 1 if equal bboxes
    """
    intersection = f_area([max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])])

    union = f_area(boxA) + f_area(boxB) - intersection
    if union == 0:
        print boxA, boxB, intersection
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


#########

def savecopy():
    for groupedDataset, groupedDatasets in getGroupedDatasets().iteritems():
        for tracker in getTrackers()[1:2]:
            label = "savedata/" + str(datetime.now().strftime("%m-%d")) + "_" + groupedDataset.replace("/", "-") + "_" + tracker
            print label
            try:
                evaluateMetricsGroup(groupedDatasets, tracker, label)

            except Exception as err:
                sys.__stdout__.write("Error on " + label + "\n" + str(err) + "\n")


def graphGlobal():
    with open("global.txt", "w") as global_file:
        global_file.write("dataset frame id xmin ymin xmax ymax\n")
        data = {}
        for dataset_name, dataset in getGroupedDatasets().iteritems():
            data[dataset_name] = {}
            for detector in [1, 5, 10]:
                label = "savedata/global_" + dataset_name.replace("/", "-") + "_" + str(detector)
                try:
                    data[dataset_name][detector] = {}

                    motpAll, motaAll = evaluateMetricsGroup(dataset, 'KCF', toFile=label + "_all", detector=detector)
                    data[dataset_name][detector]['motpAll'] = motpAll
                    data[dataset_name][detector]['motaAll'] = motaAll

                    motpsOne = []
                    motasOne = []
                    for i, dataset_element in enumerate(dataset):
                        motpOne, motaOne = evaluateMetricsGroup([dataset_element], 'KCF', toFile=label + "_one" + str(i), detector=detector)
                        motpsOne.append(motpOne)
                        motasOne.append(motaOne)
                    data[dataset_name][detector]['motpOne'] = sum(motpsOne) / len(motpsOne)
                    data[dataset_name][detector]['motaOne'] = sum(motasOne) / len(motasOne)
                    global_file.write("\t".join(map(str, data[dataset_name][detector].values())))
                except Exception as e:
                    print label, "-error:", e

        print data


if __name__ == '__main__':
    # evaluateMetrics("Laboratory/6p-c0", 'MEANSHIFT')
    # evaluateMetrics("Laboratory/6p-c0", 'CAMSHIFT')
    # evaluateMetrics("Basketball/match5-c2")

    # for dataset in getDatasets():
    #    print dataset
    #    evaluateMetrics(dataset)

    # V2

    # evaluateMetricsGroup(getGroupedDatasets()['Laboratory/6p'], 'KCF', detector=10)

    # savecopy()
    graphGlobal()
