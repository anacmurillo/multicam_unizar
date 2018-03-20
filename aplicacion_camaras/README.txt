App to view and save syncronized camera topics.
Currently supports rgb and depth, but probably more 'image' topics

How to configure:
  Add a 'settings.ini' file in the same directory as the AbelApp.py file with the folowing format:
  specify tag: [topics]
  specify cameras: <name identifier>:<topic> where <name identifier> will be shown in the dropdown menu, can be anything; and <topic> should be an active topic
  You can add comments precedding them with #
  Example file:
        [topics]
        Kinect RGB:/kinect/rgb/image_raw
        Kinect Depth:/kinect/depth/image_raw
        Orbecc RGB:/orbbec/rgb/image_raw
        Orbecc Depth:/orbbec/depth/image_raw
        test RGB:/camera/rgb/image_raw
        test Depth:/camera/depth/image_raw
  
How to run:
  $phyton AbelApp.py [dir]
  execute the command where [dir] is the path where the photos will be saved (if none specified 'default' will be used), write just a name (no spaces) to specify a subfolder on the current directory.
  For example if you want to save the photos under a 'photos' folder execute:
  $python AbelApp.py photos

How to use:
  Open the app, a roscore is launched automatically.
  Choose one or more topics from the list to toggle them, a preview will be added.
      If the topic is disabled no preview will be shown and won't be saved. In the list it will show "[Off->On]" and clicking will enable it.
      If the topic is enabled a preview will be shown and will be saved. In the list it will show "[On->Off]" and clicking will disable it.
  Change the number to modify the 'Time Window'
  Press the 'save once' button to save one photo (of each active topic).
  Press the 'save all' button to save continuously, press again to stop.
  Press 'record bag' to start recording of active topics, press again to stop.
  Press 'launch Openni/Kinect' to start a process of openni or kinect repectively (shortcut to avoid using other console).
  If one topic is active, press 'calibrate' to start the calibration process. The grid will be displayed if found, when ready click the button again to show parameters.
  
  Close the window to stop.


Tested on Ubuntu 16.04 with the folowing dependencies:
    roslib(1.13.6), rospy(1.12.12) and message_filters(1.12.12) [ROS]
    wxPython(3.0.2.0) [wxPython]
    cv-bridge(1.12.7) [OpenCV]
    numpy(1.11.0) and numpngw(0.0.6) (Numpy)
    
