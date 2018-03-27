"""
Test of tracker. Demo script.
No usage
"""
import sys

import cv2

from colorUtility import getColors

filename = "/home/jaguilar/Abel/epfl/dataset/CVLAB/Basketball/match5-c0.avi"

(major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

if __name__ == '__main__':

    # Set up trackers.
    # Instead of MIL, you can also use
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW']  # , 'GOTURN'

    trackers = {}
    colors_list = getColors(len(tracker_types))

    for i, tracker_type in enumerate(tracker_types):

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
        trackers[tracker_type] = [tracker, False, [], colors_list[i]]  # tracker, found, bbox, color

    # Read video
    video = cv2.VideoCapture(filename)

    # Exit if video not opened.
    if not video.isOpened():
        print "Could not open video"
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print "Cannot read video file"
        sys.exit()

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    # Initialize trackers with first frame and bounding box
    for tracker_type in tracker_types:
        if trackers[tracker_type][0].init(frame, bbox):
            print "initialized tracker", tracker_type, "correctly"
        else:
            print "can't initialize tracker", tracker_type

    while ok:

        # Start timer
        timer = cv2.getTickCount()

        # Update trackers
        for tracker_type in tracker_types:
            trackers[tracker_type][1], trackers[tracker_type][2] = trackers[tracker_type][0].update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding boxes
        for tracker_type in tracker_types:
            tracker, ok, bbox, color = trackers[tracker_type]
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, color, 2, 1)
                cv2.putText(frame, tracker_type, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
            # else :
            # Tracking failure
            # cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display FPS on frame
        # cv2.putText(frame, "FPS : " + str(int(fps)), (0,0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

        # Read a new frame
        ok, frame = video.read()
