#!/usr/bin/env python
from __future__ import print_function

import sys, os
import rospy
from std_msgs.msg import String, Header
from geometry_msgs.msg import PoseStamped
from matplotlib import pyplot as plt
import numpy as np
import math
import tf

#########################################################
# CONFIG AND GLOBAL VARIABLES
#########################################################

# TODO: Populate the config dictionary with any
# configuration parameters that you need
config = {
    'topic_name': "/orb_slam2_stereo/pose", 
    'node_name': "pose_subscriber",
    'sub_freq': 5,      # measurements to skip
    'pub_rate': 5,      # Hz
}   

idx = 0
measurement = None
prev_meas_x = 0
dt = 1/10 # based on Hz from dataset

#########################################################
# HELPER FUNCIONS
#########################################################

# Converts rotation matrix to euler angles
def mat2euler(M):
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > 1e-4: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return z, y, x

# Converts quaternions to rotation matrix
def quat2mat(q):
    assert len(q) == 4, "Not a valid quaternion"
    if np.linalg.norm(q) != 1.:
        q = q / np.linalg.norm(q)
    mat = np.zeros((3,3))
    mat[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    mat[0, 1] = 2*q[1]*q[2] - 2*q[3]*q[0]
    mat[0, 2] = 2*q[1]*q[3] + 2*q[2]*q[0]
    mat[1, 0] = 2*q[1]*q[2] + 2*q[3]*q[0]
    mat[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    mat[1, 2] = 2*q[2]*q[3] - 2*q[1]*q[0]
    mat[2, 0] = 2*q[1]*q[3] - 2*q[2]*q[0]
    mat[2, 1] = 2*q[2]*q[3] + 2*q[1]*q[0]
    mat[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2
    return mat

def quat2euler(q):
    return mat2euler(quat2mat(q))

def euler2vec(state):
    theta = state[3]
    phi = state[4]
    psi = state[5]

    # theta = roll (rotation around x)
    # phi = pitch (rotation around y)
    # psi = yaw (rotation around z)

    # assume x-body is forward and we are referencing to the original pose 
    x = np.cos(psi)*np.cos(phi)
    y = np.sin(psi)*np.cos(phi)
    z = np.sin(phi)

    res = np.zeros((3,1))
    res[0,0] = x
    res[1,0] = y
    res[2,0] = z
    return res



###############################################################
# KALMAN FILTER
###############################################################

class KalmanFilter(object):

    def __init__(self, dim_x, dim_y):
        # TODO: modify depending on the tracked state
        # You may add additional methods to this class
        # for building ROS messages

        self.dim_x = dim_x          # state dims
        self.dim_y = dim_y          # measurement dims

        self.x = np.zeros((dim_x,1))
        # self.x = pose       # state
        self.P = np.eye(dim_x)              # covariance
        self.Q = np.eye(dim_x)              # propagation noise

        self.F = np.eye(dim_x)              # state transition matrix
        
        self.H = np.zeros((dim_y, dim_x))   # measurement matrix
        for i in range(dim_x-1):
            self.H[i,i] = 1
        
        self.R = np.eye(dim_y)              # measurement uncertainty

    def predict(self):
        # TODO: add predict step code
        direction = euler2vec(self.x)*dt
        self.F[0, 6] = direction[0,0]
        self.F[1, 6] = direction[1,0]
        self.F[2, 6] = direction[2,0]
        self.x = np.matmul(self.F,self.x)
        self.P = np.matmul(np.matmul(self.F, self.P), np.transpose(self.F)) + self.Q
        pass

    def update(self, y):
        # TODO: add update step code. y is a vector
        # containing all the measurements
        t = self.R + np.matmul(np.matmul(self.H, self.P), np.transpose(self.H))
        K = np.matmul(np.matmul(self.P, np.transpose(self.H)), np.linalg.inv(t))
        self.x = self.x + np.matmul(K, y-np.matmul(self.H, self.x))
        self.P = np.matmul((np.eye(self.dim_x)-np.matmul(K,self.H)), self.P)
        pass

kf = KalmanFilter(1, 1)

####################################################################
# ROS SUBSCRIBER
####################################################################

def callback(data):
    global idx, config, measurement
    rospy.loginfo(rospy.get_caller_id()+"   "+str(idx))
    idx += 1
    # TODO: add code to read position and orientation from Pose
    # message, add noise and pass to measurement global variable  
    x = data.pose.position.x
    y = data.pose.position.y
    z = data.pose.position.z
    q0 = data.pose.orientation.w
    q1 = data.pose.orientation.x
    q2 = data.pose.orientation.y
    q3 = data.pose.orientation.z

    q = np.array([q0, q1, q2, q3])
    mat = quat2euler(q)
    measurement = np.array([x, y, z, mat[2], mat[1], mat[0]])

    # Uniform noise between 0, 1
    # measurement = measurement + 0.02*np.random.rand(6)

    # Gaussian with mean 0 and covariance 1
    #measurement = measurement + 0.02*np.random.randn(6)

    # Exponential distribution
    measurement = measurement + np.random.exponential(1.0, 6)

    # reshape measurement for the update of the Kalman Filter
    measurement = measurement.reshape(-1,1)

    pass

kf_states = np.zeros((3,1))
orb_slam_states = np.zeros((3,1))


def subscribe(config):
    global idx, measurement, kf_states, orb_slam_states
    # TODO: add code for creating a publisher for output 
    rospy.Subscriber(config['topic_name'], PoseStamped, callback)
    
    # measurement[6] = (measurement[0] - prev_meas_x)/dt
    # prev_meas_x = measurement[0]

    # TODO: replace with publisher loop which calls predict/update 
    # in Kalman filter and publishes the estimated pose
    kalman = KalmanFilter(7,6)


    pub = rospy.Publisher('/kalman_filter/pose', PoseStamped, queue_size=10)
    rospy.init_node(config['node_name'], anonymous=True)
    rate = rospy.Rate(config['pub_rate'])

    while not rospy.is_shutdown():
        
        if measurement is None:
                pass
        else:
            # print(measurement)
            kalman.predict()
            kalman.update(measurement)
            message = PoseStamped()
            message.header = Header()
            message.pose.position.x = kalman.x[0,0]
            message.pose.position.y = kalman.x[1,0]
            message.pose.position.z = kalman.x[2,0]
            message.pose.orientation.w = kalman.x[3,0]
            message.pose.orientation.x = kalman.x[4,0]
            message.pose.orientation.y = kalman.x[5,0]
            message.pose.orientation.z = kalman.x[6,0]
                
            pub.publish(message)
            print(kalman.x[0])
            print(kalman.x[1])
            print(kalman.x[2])
            kf_states = np.append(kf_states, kalman.x[0:3], axis=1)
            orb_slam_states = np.append(orb_slam_states, measurement[0:3], axis=1)

        rate.sleep()
    
    print(1)
    print(kf_states)
    ax = plt.axes()
    ax.set_ylabel("Y Position")
    ax.set_xlabel("X Position")
    ax.plot(kf_states[0,:], kf_states[1,:], label='kalman filter output')
    ax.plot(orb_slam_states[0,:], orb_slam_states[1,:], label='orb_slam noisy output')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    try:
        subscribe(config)
        
    except rospy.ROSInterruptException:
        pass