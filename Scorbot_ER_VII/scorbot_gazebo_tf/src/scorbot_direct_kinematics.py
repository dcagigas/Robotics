#!/usr/bin/env python

import math
import rospy
from std_msgs.msg import Float64
import roslib 
from control_msgs.msg import JointControllerState
from geometry_msgs.msg import Point


# Kinematics based on techreport "Drawing using the Scorbot-ER VII Manipulator Arm"  
# by Luke Cole, Adam Ferenc Nagy-Sochacki and Jonathan Symonds (2007)
# See: https://www.lukecole.name/doc/reports/drawing_using_the_scorbot_manipulator_arm.pdf



# (θ1,θ2,θ3,θ4) Input angles are expressed in radians
# (x,y,z) output position is expressed in millimeters
def direct_kinematics_position (theta1, theta2, theta3, theta4):
    # Distance are expressed in meters.
    d1 = 0.358  # Distance from base center (0,0,0) rotation (1) to shoulder/body center
    #d2 = -0.075      # ?? Distance from center of the base to center of the shoulder/body axis
    d2 = 0
    a1 = 0.050     # Distance from shoulder/body center to shoulder/body joint (2)
    a2 = 0.300    # Distance from shoulder/body joint to elbow/arm joint (3)
    a3 = 0.250    # Distance from elbow/arm joint to pitch/forearm joint (4)
    a4 = 0.212    # End efector (gripper) length 

    # theta1 (θ1) = base rotation angle (1)
    # theta2 (θ2) = shoulder/body rotation angle (2)
    # theta3 (θ3) = elbow/arm rotation angle (3)
    # theta4 (θ4) = pitch/forearm rotation angle (4)

    c1 = math.cos (theta1)
    c2 = math.cos (theta2)
    c23 = math.cos (theta2 + theta3)
    c234 = math.cos (theta2 + theta3 + theta4)
    s1 = math.sin (theta1)

    # x:
    # T0-5:
    x = a4*c1*c234 + c1*c23*a3 + c1*c2*a2 + c1*a1 - s1*d2
    # T0-4:
    #x = c1*c23*a3 + c1*c2*a2 + c1*a1 - s1*d2
    # T0-3:
    #x = c1*c2*a2 + c1*a1 - s1*d2
     
    # y:
    # T0-5:
    y = a4*c234*s1 + s1*c23*a3 + s1*c2*a2 + s1*a1 + c1*d2 
    # T0-4:
    #y = s1*c23*a3 + s1*c2*a2 + s1*a1 + c1*d2
    # T0-3:
    #y = s1*c2*a2 + s1*a1 + c1*d2

    s2 = math.sin (theta2)
    s23 = math.sin (theta2 + theta3)
    s234 = math.sin (theta2 + theta3 + theta4)

    # z:
    #z = s23*a3 + s2*a2 + d1
    # T0-5:
    z = -a4*s234 -s23*a3 - s2*a2 + d1
    # T0-4:
    #z = -s23*a3 - s2*a2 + d1
    # T0-3:
    #z = - s2*a2 + d1

    result = []
    result.append(x)
    result.append(y)
    result.append(z)
    return result


# It is needed the rotation matrix for obtaining the yaw, pitch and roll angles (Euler angles).
# Rotation is the same for the wrist and the scorbot end effector.
def direct_kinematics_orientation (theta1, theta2, theta3, theta4):
    # r11 r12 r13
    # r21 r22 r23
    # r31 r32 r33
    c1 = math.cos (theta1)
    s1 = math.sin (theta1)  
    c234 = math.cos (theta2 + theta3 + theta4)  

    result = []
    # Yaw: atan (r21/r11) --> yaw = math.atan ((s1*c234)/(c1*c234)) --> yaw = math.atan (s1/s1) --> yaw = theta1
    yaw = theta1
    # Pitch: atan (-r31 / sqrt (r32*r32 + r33*r33)) --> pitch = math.atan (s234 / sqrt (cos234*cos234)+0) --> 
    #        pitch = math.atan (s234 / cos234) --> pitch = theta2+theta3+theta4
    pitch = theta2 + theta3 + theta4
    # Roll: atan (r32/r33) --> There is no roll because r33 == 0
    roll = 0
    result.append(yaw)
    result.append(pitch)
    result.append(roll)
    return result


def move_angles_(theta1, theta2, theta3, theta4):
    pub_base = rospy.Publisher('/scorbot/base_position_controller/command', Float64, queue_size=10)
    pub_shoulder = rospy.Publisher('/scorbot/shoulder_position_controller/command', Float64, queue_size=10)
    pub_elbow = rospy.Publisher('/scorbot/elbow_position_controller/command', Float64, queue_size=10)
    pub_pitch = rospy.Publisher('/scorbot/pitch_position_controller/command', Float64, queue_size=10)
    pub_roll = rospy.Publisher('/scorbot/roll_position_controller/command', Float64, queue_size=10)
    pub_gripper_finger_left = rospy.Publisher('/scorbot/gripper_left_controller/command', Float64, queue_size=10)
    pub_gripper_finger_right = rospy.Publisher('/scorbot/gripper_right_controller/command', Float64, queue_size=10)
    # Publish angles:
    pub_base.publish (theta1)
    pub_shoulder.publish (theta2)
    pub_elbow.publish (theta3)
    pub_pitch.publish (theta4)
    pub_roll.publish (0)    # Wrist is fixed
    pub_gripper_finger_left.publish (-1)
    pub_gripper_finger_right.publish (-1)


def jointStatesCallback(msg):
  #global currentJointState
  #currentJointState = msg
  #Point m = msg
  print (msg)


def read_position ():
    #sub_base = rospy.Subscriber('/scorbot/base_position_controller/command', JointControllerState, self.get_joint_position)
    sub = rospy.Subscriber("/scorbot/joint_state_controller", JointControllerState, jointStatesCallback)
    #print (sub_base)
    sub_shoulder = rospy.Subscriber('/scorbot/shoulder_position_controller/command', JointControllerState, self.get_joint_position)
    #print (sub_shoulder)



if __name__ == '__main__':

    try:
        a1 = 0  #x0 = 0.8119999999   # 0.812 gives error (singular point: working area limit)
        a2 = 0  #y0 = 0
        a3 = 0  #z0 = 0.358
        a4 = 0
                                    # 01 solution
        b1 = 1                      #x1 = 0.750
        b2 = -0.5401742497052632    #y1 = 0
        b3 = -0.9355365085259083     #z1 = 0.300
        b4 = 0.39536225882064513   

                                    # 02 solution
        #b1 = 0                      #x1 = 0.750
        #b2 = 0.6616072627919588    #y1 = 0
        #b3 = 1.1324677292534087     #z1 = 0.300
        #b4 = -0.2232786652504709   

        

        print ("Begin example ...")

        data0 = direct_kinematics_position (a1, a2,a3,a4)
        print ( 'Scorbot end-effector (x,y,z) position for angles: theta1: ' + str(a1) + ' theta2: ' + str(a2) + ' theta3: ' + str(a3) + ' theta4: ' + str(a4))
        print (data0)
        data0 = direct_kinematics_orientation (a1, a2,a3,a4)
        print ( 'Scorbot end-effector (x,y,z) orientation for angles: theta1: ' + str(a1) + ' theta2: ' + str(a2) + ' theta3: ' + str(a3) + ' theta4: ' + str(a4))
        print (data0)
        print ()

        data1 = direct_kinematics_position (b1, b2, b3, b4)
        print ( 'Scorbot end-effector (x,y,z) position for angles: theta1: ' + str(b1) + ' theta2: ' + str(b2) + ' theta3: ' + str(b3) + ' theta4: ' + str(b4) )
        print (data1)
        data1 = direct_kinematics_orientation (b1, b2, b3, b4)
        print ( 'Scorbot end-effector (x,y,z) orientation for angles: theta1: ' + str(b1) + ' theta2: ' + str(b2) + ' theta3: ' + str(b3) + ' theta4: ' + str(b4))
        print (data1)
        print ()        

        rospy.init_node('simple_angle_mover')
        rate = rospy.Rate(0.2)

        while not rospy.is_shutdown():
            move_angles_ (a1,a2,-a3,-a4)
            rate.sleep()
            #read_position()
            move_angles_ (b1,b2,-b3,-b4) 
            rate.sleep()  
            #read_position()                    

    except rospy.ROSInterruptException:
        pass


