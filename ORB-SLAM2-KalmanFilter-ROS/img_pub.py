#!/usr/bin/env python
from __future__ import print_function

import sys, os
import rospy
import cv2
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

#########################################################
# CONFIG AND GLOBAL VARIABLES
#########################################################

# TODO: Populate the config dictionary with any
# configuration parameters that you need
config = {
    'topic_name_img_left': "/cam_front/left/image_rect_color", 
    'topic_name_info_left': "/image_left/camera_info", 
    'topic_name_img_right': "/cam_front/right/image_rect_color", 
    'topic_name_info_right': "image_right/camera_info", 
    'node_name': "stereo_image_publisher",
    'pub_rate': 2,   # Hz
    'data_dir': "/home/ros-industrial/Desktop/2011_09_26_drive_0002_sync",
    'img_dims': (512, 1392)
}

#########################################################
# DATASET
#########################################################

class ImgDataset:
    def __init__(self, config):
        self.data_dir = config['data_dir']
        self.files = sorted(os.listdir(self.data_dir+'/image_02/data/'))
        self.count = 0
        self.bridge = CvBridge()
        self.img_dims = config['img_dims']

    def build_camera_images(self):
        if self.count >= len(self.files):
            return None
        else:
            # TODO: Read stereo camera images from the dataset and 
            # convert to ROS messages. Messages for both left and right
            # camera images must be returned.  
            lm = cv2.imread(self.data_dir + "/image_02/data/" + self.files[self.count])
            rm = cv2.imread(self.data_dir + "/image_03/data/" + self.files[self.count])
            # print(self.data_dir + "/image02/data/" + self.files[0])
            
            
            img = Image()
            lmsg = Image()
            lmsg.height = config["img_dims"][0]
            lmsg.width = config["img_dims"][1]
            lmsg.encoding = "rgb8"
            lmsg = self.bridge.cv2_to_imgmsg(lm, "rgb8")

            rmsg = Image()
            rmsg.height = config["img_dims"][0]
            rmsg.width = config["img_dims"][1]
            rmsg.encoding = "rgb8"
            rmsg = self.bridge.cv2_to_imgmsg(rm, "rgb8")


            self.count += 1
            return lmsg, rmsg

    def build_camera_info(self):
        # TODO: Build and return camera calibration messages
        # for both left and right cameras 
        lcamera_info = CameraInfo()
        lcamera_info.header = Header()
        lcamera_info.header.stamp = rospy.Time.now()
        lcamera_info.width = config["img_dims"][1]
        lcamera_info.height = config["img_dims"][0]
        lcamera_info.K = [9.597910e+02, 0.000000e+00, 6.960217e+02, 0.000000e+00, 9.569251e+02, 2.241806e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00]
        lcamera_info.R = [9.998817e-01, 1.511453e-02, -2.841595e-03, -1.511724e-02, 9.998853e-01, -9.338510e-04, 2.827154e-03, 9.766976e-04, 9.999955e-01]
        lcamera_info.P = [7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01, 0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01, 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]
        lcamera_info.D = [-3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02]
        lcamera_info.distortion_model = "plumb_bob"

        rcamera_info = CameraInfo()
        rcamera_info.header.stamp = rospy.Time.now()
        rcamera_info.header = Header()
        rcamera_info.width = config["img_dims"][1]
        rcamera_info.height = config["img_dims"][0]
        rcamera_info.K = [9.597910e+02, 0.000000e+00, 6.960217e+02, 0.000000e+00, 9.569251e+02, 2.241806e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00]
        rcamera_info.R = [9.998817e-01, 1.511453e-02, -2.841595e-03, -1.511724e-02, 9.998853e-01, -9.338510e-04, 2.827154e-03, 9.766976e-04, 9.999955e-01]
        rcamera_info.P = [7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01, 0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01, 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]
        rcamera_info.D = [-3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02]
        rcamera_info.distortion_model = "plumb_bob"

        return lcamera_info, rcamera_info 
    

#########################################################
# ROS PUBLISHER
#########################################################

def publish(config):
    lpub = rospy.Publisher(config['topic_name_img_left'], Image, queue_size=10)
    lpub_info = rospy.Publisher(config['topic_name_info_left'], CameraInfo, queue_size=10)
    rpub = rospy.Publisher(config['topic_name_img_right'], Image, queue_size=10)
    rpub_info = rospy.Publisher(config['topic_name_info_right'], CameraInfo, queue_size=10)
    
    rospy.init_node(config['node_name'], anonymous=True)
    rate = rospy.Rate(config['pub_rate'])

    dataset = ImgDataset(config)

    while not rospy.is_shutdown():
        try:
            lmsg, rmsg = dataset.build_camera_images()
            lmsg_info, rmsg_info = dataset.build_camera_info()
            lpub.publish(lmsg)
            rpub.publish(rmsg)
            lpub_info.publish(lmsg_info)
            rpub_info.publish(rmsg_info)
            rospy.loginfo("Published Images {id} and Info".format(id=dataset.count))
        except CvBridgeError as e:
            print(e)
        
        rate.sleep()

if __name__ == '__main__':
    try:
        publish(config)
    except rospy.ROSInterruptException:
        pass