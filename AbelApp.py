#!/usr/bin/env python
"""
new app, using wxPython for gui
usage:
python <name>.py [folderPath/Name]

abel
"""
import roslib
roslib.load_manifest('rospy')
roslib.load_manifest('sensor_msgs')

import rospy
from sensor_msgs.msg import Image

import message_filters
import os

import wx

import sys

import cv2
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import numpngw

import ConfigParser

import subprocess #to generate a subprocess

from time import sleep #to sleep

#settings
MAX_SCALE = 0.25
#/settings

#globalvariables
bridge = CvBridge()

#The file utilities
class FileUtilities():
    def __init__(self,path_base):
    
        #file things
        self.path_base = path_base
        if not os.path.exists(self.path_base):
          os.makedirs(self.path_base)
        self.intR = 0000
        self.file = open(self.path_base+"/List.txt",'w')
        
        #cv things
        self.bridge = CvBridge()
        
        # What we do during shutdown
        rospy.on_shutdown(self.cleanup)
        
        
    def image_save(self, callback, *streams):

        #pre: prepare
        self.file.write("Time:"+ rospy.get_time().__str__()+"\tFrame:"+str(self.intR))
        time = rospy.get_time().__str__()
        path=self.path_base+"/"
        
        #write files
        for i,stream in enumerate(streams):
            indx=str(self.intR)+"_s"+str(i)
            # Use cv_bridge() to convert the ROS image to OpenCV format
            try:
              if stream.encoding in ['rgba8','rgb8','bgr8']:
                  rgb = self.bridge.imgmsg_to_cv2(stream, stream.encoding)
                  cv2.imwrite(path + "/Frame_" + indx + "_RGB.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
              elif stream.encoding == '16UC1':
                  depth = self.bridge.imgmsg_to_cv2(stream, "passthrough")
                  
                  depth_array = np.array(depth*1., dtype=np.uint16)
                  depth_array = np.nan_to_num(depth_array)
                  
                  outputImg8U = cv2.convertScaleAbs(depth_array, alpha=(255.0/65535.0))
                  outputImg8U = np.array(outputImg8U, dtype=np.uint8)

                  numpngw.write_png(path + "/Frame_" + indx + "_Depth.png", depth_array)

                  cv2.imwrite(path + "/Frame_" + indx + "_Depth_8U.jpg", outputImg8U)
              elif stream.encoding == 'mono16':
                  im = self.bridge.imgmsg_to_cv2(stream, stream.encoding)
                  cv2.imwrite(path+"/Frame_"+indx+"_IR.png", im)
              else:
                  print("unknown encoding to save: "+stream.encoding)

            except CvBridgeError, e:
                print e
          
        #post: closing
        self.intR += 1
        self.file.write("\n")
        print("saved frame "+str(self.intR))
        wx.CallAfter(callback)
        
    def cleanup(self):
        #print "Shutting down vision node."
        self.file.close()



#calibration utilities
class CalibrationUtils():
    def __init__(self):
        self.activeInt = False
        self.activeExt = False
        
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.gridSize = (10,7) #one less of real size
        
    def startInt(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.gridSize[0]*self.gridSize[1],3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.gridSize[0],0:self.gridSize[1]].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.
        self.imageSize = None
        self.activeInt = True
        self.avgs = []
        
    def stopInt(self):
        self.activeInt = False
        if self.imageSize != None:
            print("calibrating...")
            wx.CallAfter(self.calibrate)
    def calibrate(self):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.imageSize,None,None)
        print(ret)
        print(mtx)
        print(dist)
        print(rvecs)
        print(tvecs)
        mean_error = 0
        for i in xrange(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(self.imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        
        print "total error: ", mean_error/len(self.objpoints)
    
    #calibrates rgb image adding chessboard overlay and adding the calibration to the list if different enough
    def calibImage(self,rgb):
        if self.activeInt:
            gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.gridSize,None)
            self.imageSize = gray.shape[::-1]
            
            # If found, add object points, image points (after refining them)
            if ret == True:
                # Draw and display the corners
                cv2.drawChessboardCorners(rgb, self.gridSize, corners,ret)
                
                averagePoint = [self.imageSize[0],self.imageSize[1]]
                for p in corners:
                    averagePoint[0] = (averagePoint[0]+p[0][0]) /2
                    averagePoint[1] = (averagePoint[1]+p[0][1]) /2
                
                #add only if different enough
                if len(self.avgs) == 0 or min([ abs(averagePoint[0]-p[0])+abs(averagePoint[1]-p[1]) for p in self.avgs] ) > 50:
                    self.avgs.append(averagePoint)
                    print(len(self.avgs))
                    self.objpoints.append(self.objp)
                    cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
                    self.imgpoints.append(corners)

            self.imageSize = gray.shape[::-1]
            


#The app
class ImageViewApp(wx.App):
    def OnExit(self):
        rospy.signal_shutdown("App closed")
        self.files.cleanup()
        if not self.recordProcess == None:
            self.recordProcess.send_signal(subprocess.signal.SIGINT)
            rospy.loginfo('topic recording stopped')
        if not self.openni2Process == None:
            self.openni2Process.send_signal(subprocess.signal.SIGTERM)#SIGINT
            rospy.loginfo('openni2 process stopped')
        if not self.kinect2Process == None:
            self.kinect2Process.send_signal(subprocess.signal.SIGTERM)#SIGINT
            rospy.loginfo('Kinect2 process stopped')

    def OnInit(self):
        #init
        self.saving = False
        self.drawing = False
        self.config = ConfigParser.ConfigParser()
        self.config.read("settings.ini")
        self.topicEnabled = []
        self.recordProcess = None
        self.calUtils = CalibrationUtils()
        self.recordProcess = None
        self.openni2Process = None
        self.kinect2Process = None
        
        
        #main frame with sizer
        self.frame = wx.Frame(None, title = "Abel Image View",style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)
        sizer = wx.BoxSizer(wx.VERTICAL) 
        self.frame.SetSizer(sizer)
        
        
        
        #top panel 1 (utilities) with buttons and sizer
        panelTopUtils = wx.Panel(self.frame)
        sizerTopUtils = wx.BoxSizer(wx.HORIZONTAL)
        panelTopUtils.SetSizer(sizerTopUtils)
        sizer.Add(panelTopUtils,0)
        
        #button to record topics
        self.recordBag = wx.ToggleButton(panelTopUtils,-1,"Record bag: Off  ")#extra spaces instead of set manually the size
        self.recordBag.Bind(wx.EVT_TOGGLEBUTTON,self.onBtnRecordBag)
        sizerTopUtils.Add(self.recordBag,1)
        
        #button to launch openni2
        self.runOpenni2 = wx.ToggleButton(panelTopUtils,-1,"Launch Openni2  ")#extra spaces instead of set manually the size
        self.runOpenni2.Bind(wx.EVT_TOGGLEBUTTON,self.onBtnRunOpenni2)
        sizerTopUtils.Add(self.runOpenni2,1)
        
        #button to launch kinect2
        self.runKinect2 = wx.ToggleButton(panelTopUtils,-1,"Launch Kinect2  ")#extra spaces instead of set manually the size
        self.runKinect2.Bind(wx.EVT_TOGGLEBUTTON,self.onBtnRunKinect2)
        sizerTopUtils.Add(self.runKinect2,1)
        
        #button to detect grid
        self.calibrateInt = wx.ToggleButton(panelTopUtils,-1,"Calibrate Intrinsic")
        self.calibrateInt.Bind(wx.EVT_TOGGLEBUTTON,self.onBtnCalibrateInt)
        sizerTopUtils.Add(self.calibrateInt,1)
        
        
        
        
        #top panel 2 (config) with buttons and sizer
        panelTopSettings = wx.Panel(self.frame)
        sizerTopSettings = wx.BoxSizer(wx.HORIZONTAL)
        panelTopSettings.SetSizer(sizerTopSettings)
        sizer.Add(panelTopSettings,0)
        
        
        #create combobox
        self.comboBox = wx.ComboBox(panelTopSettings,choices=["Not loaded"],style=wx.CB_READONLY)
        self.comboBox.Bind(wx.EVT_COMBOBOX,self.onComboBox)
        sizerTopSettings.Add(self.comboBox,0)
        self.updateComboBox()
        
        #create timeWindow editor
        self.timeWindow = wx.SpinCtrlDouble(panelTopSettings,initial=0.05,min=0,max=1,inc=0.01)
        self.timeWindow.Bind(wx.EVT_SPINCTRLDOUBLE,self.onTimeWindowChange)
        sizerTopSettings.Add(self.timeWindow)
        
        #button to save once
        self.saveOnce = wx.ToggleButton(panelTopSettings,-1,"Click to save once")
        self.saveOnce.Bind(wx.EVT_BUTTON,self.onBtnSaveOnce)
        sizerTopSettings.Add(self.saveOnce,1)
        
        #button to save all
        self.saveAll = wx.ToggleButton(panelTopSettings,-1,"Save all: Off  ")#extra spaces instead of set manually the size
        self.saveAll.Bind(wx.EVT_TOGGLEBUTTON,self.onBtnSaveAll)
        sizerTopSettings.Add(self.saveAll,1)
        
        

        #middle panel with imageviewers and sizer
        self.panelMiddle = wx.Panel(self.frame)
        self.sizerMiddle = wx.BoxSizer(wx.HORIZONTAL)
        self.panelMiddle.SetSizer(self.sizerMiddle)
        sizer.Add(self.panelMiddle,0)
        
        
        #display
        self.frame.Show()
        self.refresh()
        return True
    
    def setParameters(self,fileUtilites):
        self.files = fileUtilites
        
        
    def onClose(self, event):
        rospy.signal_shutdown("User closed the app")
        self.Close()
        
        
    
    
    #timeWindow
    def onTimeWindowChange(self,event):
        #pass
        self.changeTopics() #too many events
    
    #combobox
    def updateComboBox(self):
        names = self.config.options("topics")
        self.topicEnabled.extend([False for i in range(len(self.topicEnabled),len(names))])
        
        self.comboBox.Clear()
        self.comboBox.Append("Active: "+str(self.topicEnabled.count(True)))
        for i in range(len(names)):
            self.comboBox.Append(names[i]+"   "+("*Active*" if self.topicEnabled[i] else ""))
        self.comboBox.SetSelection(0)
            
            
    def onComboBox(self,event):#event.GetEventObject()
        sel = self.comboBox.GetSelection()
        
        if sel==0:
            self.topicEnabled = [False] * len(self.topicEnabled)
        else:
            sel = sel-1
            self.topicEnabled[sel]=not self.topicEnabled[sel]
        self.changeTopics()
        self.comboBox.Layout()
        
        
    def changeTopics(self):
        self.updateComboBox()
        #remove old subscribers and images
        if hasattr(self, 'subscribers'):
            for sub in self.subscribers:
                sub.unregister()
        self.sizerMiddle.Clear(True)
                
        #create new subscribers and images
        self.subscribers = []
        for i,enabled in enumerate(self.topicEnabled):
            if enabled:
                self.subscribers.append( message_filters.Subscriber(self.config.get("topics",self.config.options("topics")[i] ), Image) )
                imageviewer = wx.StaticBitmap(self.panelMiddle)
                self.sizerMiddle.Add(imageviewer,0)
        
        if len(self.subscribers)>0:
            self.synchronizer = message_filters.ApproximateTimeSynchronizer( self.subscribers, 1000,self.timeWindow.GetValue())
            self.synchronizer.registerCallback(self.image_callback_all)
        self.refresh()
        
        
        
    #buttons
    def onBtnSaveOnce(self,event):
        self.saveOnce.SetLabel("Ready")
        
    def onBtnSaveAll(self,event):
        state = self.saveAll.GetValue()
        if state:
            self.saveAll.SetLabel("Save all : On")
            self.saveOnce.Enable(False)
        else:
            self.saveAll.SetLabel("Save all : Off")
            self.saveOnce.Enable(True)
    
    def onBtnRecordBag(self,event):
        state = self.recordBag.GetValue()
        if state:
            if self.topicEnabled.count(True)!=0:
                command = "rosbag record"
                for i,enabled in enumerate(self.topicEnabled):
                    if enabled:
                        command = command+" "+self.config.get("topics",self.config.options("topics")[i])
                self.recordProcess = subprocess.Popen(command, cwd=self.files.path_base, shell=True)
                self.recordBag.SetLabel("Record bag: On")
            else:
                print("no active topics")
                self.recordBag.SetValue(False)
        else:
            if not self.recordProcess == None:
                self.recordProcess.send_signal(subprocess.signal.SIGINT)
                self.recordProcess = None
                rospy.loginfo('topic recording stopped')
            else:
                print("no record to stop")
            self.recordBag.SetLabel("Record bag: Off")
            
    def onBtnRunOpenni2(self,event):
        state = self.runOpenni2.GetValue()
        if(state):
            self.openni2Process = subprocess.Popen("exec roslaunch openni2_launch openni2.launch", cwd=self.files.path_base, shell=True)
            self.runOpenni2.SetLabel("Openni2 *active*")
        else:
            if not self.openni2Process == None:
                self.openni2Process.send_signal(subprocess.signal.SIGTERM)#SIGINT
                self.openni2Process = None
                rospy.loginfo('openni2 process stopped')
            else:
                print("no record to stop")
            self.runOpenni2.SetLabel("Launch Openni2  ")
            
    def onBtnRunKinect2(self,event):
        state = self.runKinect2.GetValue()
        if(state):
            self.kinect2Process = subprocess.Popen("source /home/jaguilar/catkin_ws/devel/setup.bash && exec roslaunch kinect2_bridge kinect2_bridge.launch", cwd=self.files.path_base, shell=True,executable="/bin/bash")
            self.runKinect2.SetLabel("Kinect2 *active*")
        else:
            if not self.kinect2Process == None:
                self.kinect2Process.send_signal(subprocess.signal.SIGTERM)#SIGINT
                self.kinect2Process = None
                rospy.loginfo('Kinect2 process stopped')
            else:
                print("no record to stop")
            self.runKinect2.SetLabel("Launch Kinect2  ")
    
    def onBtnCalibrateInt(self,event):
        state = self.calibrateInt.GetValue()
        if(state and len(self.sizerMiddle.GetChildren())==1):
            self.calUtils.startInt()
        else:
            self.calUtils.stopInt()
            self.runKinect2.SetValue(False)
        
    
    
    #images
    def image_callback_all(self, *images):
        if self.saving or self.drawing: #we can remove self.saving, but this way we know what is being saved
            return
        
        # make sure we update in the UI thread
        self.drawing=True
        wx.CallAfter(self.update, *images)
        # http://wiki.wxpython.org/LongRunningTasks
        
        if self.saveOnce.GetValue() or self.saveAll.GetValue():
            self.saving = True
            self.saveOnce.SetLabel("Saving...")
            wx.CallAfter(self.files.image_save,self.saveEnd,*images)
        
        
    def saveEnd(self):
        self.saving = False
        self.saveOnce.SetLabel("Saved")
        self.saveOnce.SetValue(False)
        
        
    def update(self, *images):
        #not the same number of streams and displays, cancel (user added/removed one)
        if len(images)!=len(self.sizerMiddle.GetChildren()):
            return
    
        for i in range(len(images)):
            self.updateViewer(self.sizerMiddle.GetItem(i).GetWindow(),images[i])
        self.drawing=False
            
    def updateViewer(self,viewer,image):
        # http://www.ros.org/doc/api/sensor_msgs/html/msg/Image.html
        
        #fix for 16UC1 images
        if image.encoding == '16UC1':
            image.encoding = 'mono16'

        rgb = bridge.imgmsg_to_cv2(image, 'rgb8')
                  
        
        self.calUtils.calibImage(rgb)
        
        #scale image to max size
        sizeX = image.width
        sizeY = image.height
        scale = MAX_SCALE*max((0.0+wx.GetDisplaySize().GetWidth())/sizeX,(0.0+wx.GetDisplaySize().GetHeight())/sizeY)
        if scale<1:
            sizeX = int(sizeX*scale)
            sizeY = int(sizeY*scale)
            rgb = cv2.resize(rgb, (sizeX, sizeY)) 
        bmp = wx.BitmapFromBuffer(sizeX, sizeY, rgb)
        
        
        #set and update if neccesary
        empty = not viewer.GetBitmap().IsOk()
        
        if bmp!=None:
            viewer.SetBitmap(bmp)
            viewer.Layout()
        
        if empty:
            self.refresh()
            print(image.encoding)
            
    def refresh(self):
        self.frame.Layout()
        self.frame.Fit()
        self.frame.Center()
        self.frame.Update()
        self.frame.Refresh()
        print "reload"
        
        
            

def main(argv):
    print(__doc__)
    roscore = subprocess.Popen('exec roscore',shell=True)
    files = FileUtilities(argv[1])
    app = ImageViewApp()
    app.setParameters(files)
    rospy.init_node('AbelApp',disable_signals=True)
    
    app.MainLoop()
    
    print("Shutting down app...")
    roscore.send_signal(subprocess.signal.SIGTERM)#SIGINT
    sleep(1)
    return 0

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("No folder specified, using 'default'")
        sys.argv.append('default')
    
    sys.exit(main(sys.argv))
