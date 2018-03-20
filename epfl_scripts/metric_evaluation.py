from groundTruthParser import parseFile, getFilenames
from evaluator import evalFile, getTrackerName

import matplotlib.pyplot as plt
import math

precision_range = 100


def evaluate(filename):
    track_ids, data_groundTruth = parseFile(filename)
    
    data_tracker, n_frames = evalFile(filename)
    
    motp_total, motp_ids = motp(track_ids, n_frames, data_groundTruth, data_tracker)
    print "MOTP: total =", motp_total
    for id in track_ids:
        print "    ", id, "=", motp_ids[id]
    
    mota_total, mota_ids = mota(track_ids, n_frames, data_groundTruth, data_tracker)
    print "MOTA: total =", mota_total
    for id in track_ids:
        print "    ", id, "=", mota_ids[id]
    
    x = []
    y = []
    c = []
    for threshold in [ i*1./precision_range for i in xrange(0,precision_range+1)]:
        precision, recall = precision_recall(track_ids, n_frames, data_groundTruth, data_tracker, threshold)
        y.append(precision)
        x.append(recall)
        c.append(threshold)
        #plt.annotate(threshold, xy=(recall,precision))#,xytext=(0, threshold))
    plt.colorbar(
        plt.scatter(x,y,c=c,edgecolors='none')
    )
    plt.xlabel('recall')
    plt.xlim([0,1])
    plt.ylabel('precision')
    plt.ylim([0,1])
    plt.title('precision-recall - ' + filename + ' - ' + getTrackerName())
    plt.show()
    
    
    
    
def precision_recall(ids, frames, groundtruth, tracker, threshold):

    true_negative  = 0. #person not in groundtruth and not found by tracker
    false_positive = 0. #tracker found person not in groundtruth
    false_negative = 0. #person in groundtruth not found by tracker
    true_positive  = 0. #tracker and groundtruth found the same person

    for id in ids:
        for frame in range(frames):
            xmin, ymin, xmax, ymax, lost, occluded, generated, label = groundtruth[frame][id]
            bbox_gt = None if lost else [xmin, ymin, xmax, ymax]
            bbox_tr = tracker[frame].get(id,None)
            
            if bbox_gt == None and bbox_tr == None:
                true_negative += 1
            elif bbox_gt == None and bbox_tr != None:
                false_positive += 1
            elif bbox_gt != None and bbox_tr == None:
                false_negative += 1
            else:
                if getSimilarity(bbox_gt, bbox_tr) >= threshold:
                    true_positive += 1
                else:
                    false_negative += 1 #false_negative or false_positive??
                    
    #print true_negative, false_positive, false_negative, true_positive
    
    return true_positive / ( true_positive + false_positive ) , true_positive / ( true_positive + false_negative )


def mota(ids, frames, groundtruth, tracker):
    persons = {}
    total = [0., 0., 0., 0.]

    for id in ids:
        mt  = 0. #number of misses (persons not found)
        fpt = 0. #number of false positives (persons found but not in groundtruth)
        mme = 0. #number of mismatches (persons found but not from this groundtruth)
        gt  = 0. #number of goundtruth available
        for frame in range(frames):
            xmin, ymin, xmax, ymax, lost, occluded, generated, label = groundtruth[frame][id]
            bbox_gt = None if lost else [xmin, ymin, xmax, ymax]
            bbox_tr = tracker[frame].get(id,None)
            
            if bbox_gt != None:
                gt += 1
            
            if bbox_gt == None and bbox_tr == None:
                None
            elif bbox_gt == None and bbox_tr != None:
                fpt += 1
            elif bbox_gt != None and bbox_tr == None:
                mt += 1
            else:
                if getSimilarity(bbox_gt, bbox_tr) >= 0.5: #magic number, consider change with distance
                    None
                else:
                    mme += 1
        persons[id] = 1. - (mt + fpt + mme) / gt
        for i in range(4): total[i] += [mt, fpt, mme, gt][i];
    
    return 1. - (total[0] + total[1] + total[2]) / total[3] , persons
                
#using different evaluation
def motp(ids, frames, groundtruth, tracker):
    persons = {}
    total = [0., 0.]
    
    for id in ids:
        distance = 0.
        matches  = 0.
        for frame in range(frames):
            xmin, ymin, xmax, ymax, lost, occluded, generated, label = groundtruth[frame][id]
            bbox_gt = None if lost else [xmin, ymin, xmax, ymax]
            bbox_tr = tracker[frame].get(id,None)
            
            if bbox_gt != None and bbox_tr != None:
                distance += f_distance(bbox_gt, bbox_tr) #getSimilarity(bbox_gt, bbox_tr)
                matches += 1
        persons[id] = distance / matches
        for i in range(2): total[i] += [distance, matches][i]
        
    return total[0] / total[1] , persons



def getSimilarity(boxA, boxB):
    intersection = f_area([ max(boxA[0],boxB[0]), max(boxA[1],boxB[1]), min(boxA[2],boxB[2]) , min(boxA[3],boxB[3]) ])
    
    union = f_area(boxA) + f_area(boxB) - intersection
    
    return intersection / union
    
    
def f_area(r):
    return (r[2]-r[0])*1.*(r[3]-r[1]) if r[2]>r[0] and r[3]>r[1] else 0.;


def f_distance(boxA, boxB):
    return f_euclidian( f_center(boxA) , f_center(boxB) )

def f_center(box):
    return ( (box[2]+box[0])/2 , (box[3]+box[1])/2 )
    
def f_euclidian(a,b):
    return math.sqrt( (b[0]-a[0])**2 + (b[1]-a[1])**2 )


if __name__ == '__main__' :        
    evaluate("Laboratory/6p-c0")
    #evaluate("Basketball/match5-c2")
    
    #for filename in getFilenames():
    #    print filename
    #    evaluate(filename)
    
