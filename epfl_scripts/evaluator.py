from groundTruthParser import parseFile
from colorUtility import getColors
from cache import cachedObject

import cv2
import sys
import numpy as np #it's numpy

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

#settings
video_folder = "/home/jaguilar/Abel/epfl/dataset/CVLAB/"


tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW']#, 'GOTURN' makes error
tracker_type = tracker_types[4]

def getTrackerName():
    return tracker_type;

def getTracker():
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
    return tracker
    

def evalFile(filename, display = False):
    return cachedObject(filename+tracker_type, lambda: _evalFile(filename, display))

def _evalFile(filename, display = False):
 
    #parse groundtruth
    track_ids, data = parseFile(filename)
 
 
    colors_list = getColors(len(track_ids))
    trackers = {}
    for id in track_ids:
        trackers[id] = None
    
    
    
    # Read video
    video = cv2.VideoCapture(video_folder+filename+".avi")
 
    # Exit if video not opened.
    if not video.isOpened():
        print "Could not open video"
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print "Cannot read video file"
        sys.exit()
     
     
    #initialize detecion set 
    data_detected = {}
    
 
    frame_index = 0
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
            
            
        #parse trackers
        for id in track_ids:
            if trackers[id] == None:
                xmin, ymin, xmax, ymax, lost, occluded, generated, label = data[frame_index][id]
                if not lost:
                    #initialize tracker
                    tracker = getTracker()
                    bbox = (xmin,ymin,xmax-xmin,ymax-ymin)
                    if tracker.init(frame, bbox):
                        print "initialized tracker", tracker_type, "for person", id, "correctly"
                        trackers[id] = [tracker, True, bbox, colors_list[id]] #tracker, found, bbox, color
                    else:
                        print "can't initialize tracker", tracker_type
            else:
                # Update tracker
                trackers[id][1], trackers[id][2] = trackers[id][0].update(frame)

        
        # Evaluate bounding boxes
        data_detected[frame_index] = {}
        for id in track_ids:
            if trackers[id] != None:
                tracker, ok, bbox, color = trackers[id]
                if ok:
                    # Tracking success
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    data_detected[frame_index][id] = [p1[0], p1[1], p2[0], p2[1]]; #xmin, ymin, xmax, ymax
                    if(display):
                        cv2.rectangle(frame, p1, p2, color, 2, 1)
                        cv2.putText(frame, str(id), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
                else :
                    # Tracking failure
                    #cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                    None
         
        if(display):
            # Display result
            cv2.imshow("Tracking", frame)
     
            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break
        else:
            sys.stdout.write("\r"+ str(frame_index)+" ")
            sys.stdout.flush()
        frame_index += 1
    print ""
        
    return data_detected, frame_index
        
        
if __name__ == '__main__' :        
    #evalFile("Basketball/match5-c0")
    print evalFile("Laboratory/6p-c0",True)
