#!/usr/bin/env python

import math
import rospy
from std_msgs.msg import Float64
import roslib 
from control_msgs.msg import JointControllerState
from geometry_msgs.msg import Point

import numpy as np 
import matplotlib.pyplot as plt
import sympy as sym
from sympy import *


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



def inverse_kinematics_position (x, y, z):
    # Distance are expressed in meters.
    d1 = 0.358  # Distance from base center (0,0,0) rotation (1) to shoulder/body center
    #d2 = -0.075      # ?? Distance from center of the base to center of the shoulder/body axis
    d2 = 0
    a1 = 0.050     # Distance from shoulder/body center to shoulder/body joint (2)
    a2 = 0.300    # Distance from shoulder/body joint to elbow/arm joint (3)
    a3 = 0.250    # Distance from elbow/arm joint to pitch/forearm joint (4)
    a4 = 0.212    # End efector (gripper) length 

    #The angle of the base is computed (d2 might be included although we are not doing at this moment). 
    #The rotation angle of the base only depends on the (x,y) plane
    q1 = np.zeros(2)
    q1[0] = np.arctan2(y+d2,x)
    q1[1] = q1[0]

    #This is done to solve the three links planar arm problem formed by (x,z).
    #The transformation on the height is due to the frame of reference of the robot.
    #The transformation on 'x' is to set the projection due to the movement of the base and to consider 
    #the displacement a1. We used (x,y) for the sake of simplicity when plotting. 
    y = -z + d1
    x = x/np.cos(q1[0])-a1

    #Selection of the wrist. Since we have an infinite number of solutions for the inverse kinematics, the way to reduce the number 
    #of them is to fix the position of the wrist (wx, wy). To do that, we consider several possibilities: 

    #If the target position is on the 'y' axis, the position of the wrist is inmediatly fixed.  
    if(x == 0):
        wx = 0.0 
        wy = y-a4
    #Second case is that all the possibilities for the wirst are inside the working space of the robot without 
    # considering the last link
    elif ((abs(np.sqrt(x**2+y**2)))<= (a2+a3)):
        wx = x - a4*(np.sqrt(2)/2)
        wy = y + a4*(np.sqrt(2)/2)
    #Following is the general case where some possibilities for the wrist are inside the working space of (a2+a3) 
    #and some others are   not. 
    else: 
        tx,ty= symbols('x,y')
        circle_1 = Eq((tx-x)**2+(ty-y)**2,a4**2)
        circle_2 = Eq(tx**2+ty**2,(a2+a3)**2)
    
        try:
        
            #It could be that there is only one solution for both circles. It means that the intersection 
            #is the position for the wrist
            sol1=solve([circle_1,circle_2],(tx,ty))
            if (len(sol1) == 1): 
                (wx, wy)= sol1[0]
                wx=float(wx)
                wy=float(wy)
    
            else:             
                (tx,ty)=solve([circle_1,circle_2],(tx,ty))

                a=np.array([tx[1]-y],dtype=np.float64)
                b=np.array([tx[0]-x],dtype=np.float64)
                c1=np.arctan2(a,b)
            
                a=np.array([ty[1]-y],dtype=np.float64)
                b=np.array([ty[0]-x],dtype=np.float64)
                c2=np.arctan2(a,b)

                #We set the angle values to positive values so we can then create the incremental array 
                if (c1<0): 
                    c1=c1+2*np.pi
                elif (c2<0):
                    c2=c2+2*np.pi
    
                #Now, c1 and c2 are positive angles (anti clock-wise reference) and we can build the incremental vector of angles. 
                if (c1>c2): 
                    theta=np.arange(c2,np.pi,0.0001)
                    first_border=c2  
                else:
                    theta=np.arange(c1,np.pi,0.0001)
                    first_border=c1
    
                #We give values to the the arch which are posible solutions for the wirst. 
                x_ = x + a4*np.cos(theta)
                y_ = y + a4*np.sin(theta)

                #Then, the wrist position can be selected either by minimizing the proximity to x or the mean
                #The mean might be out of the workspace if there is intersection but the target is higher than 
                #component 'y' of the intersection
                if (y>=float(sym.re(tx[1])) or y >= float(sym.re(ty[1]))):
                    wx=max(float(sym.re(tx[0])), float(sym.re(ty[0])))
                    wy=min(abs(float(sym.re(tx[1]))), abs(float(sym.re(ty[1]))))
        
                else: 
                    wx=x_.mean()
                    wy=y_.mean()
    
        except: 
            #If there is not a real solution: 
            if ((float(sym.im(tx[0])) != 0.0) or (float(sym.im(tx[1])) != 0.0) or (float(sym.im(ty[0])) != 0.0) or (float(sym.im(ty[1])) != 0.0)): 
                print("Error: the position cannot be reached. There is not a suitable position for the wrist.")
            else:
                print("Error: the position cannot be reached")

    # The constraints are checked
    #The second restriction: 
    #print "The second restriction:"

    if (a2+a3) < (np.sqrt(wx*wx+wy*wy)):
        print ("The wirst is out of the work-space")
    else: 
        #Computation of the cos(q2)(c3)
        c3 = (wy**2+wx**2-a2**2-a3**2)/(2*a2*a3)

        #Computation of the sin(q3)(s3)
        #The sign choosen will determine the up or down elbow configuration
 
        s3 = -np.sqrt(1-c3**2)
        #Computation of q2    
        q3 = np.arctan2(s3,c3)

        q3_bis=-q3
    
        #Computation of the cos(q2)(c2)
        c2=(a3*s3*wy+wx*(a2+a3*c3))/(a2**2+2*a2*a3*c3+a3**2)
        #Computation of the sin(q2)(s2)
        s2=np.sqrt(1-c2*c2)
        s2_bis=-np.sqrt(1-c2*c2)
        #Computation of q2
        q2=np.arctan2(s2,c2)
        q2_bis=np.arctan2(s2_bis,c2)

        #Computation of the cos(q2)(c2)
        c2=(a3*(np.sin(q3_bis))*wy+wx*(a2+a3*c3))/(a2*a2+2*a2*a3*c3+a3*a3)
        #Computation of the sin(q2)(s2)
        s2_1=np.sqrt(1-c2*c2)
        s2_1_bis=-np.sqrt(1-c2*c2)
        #Computation of q2
        q2_1=np.arctan2(s2_1,c2)
        q2_1_bis=np.arctan2(s2_1_bis,c2)

        #Build arrays for each set 
        q2_set=(q2,q2_bis,q2_1,q2_1_bis)
        q3_set=(q3,q3,q3_bis,q3_bis)

        #for loop to test the direct kinematics
        k=0
        #We allocate two positions since only two valid configurations are possible (elbow up/down)
        q2=np.zeros(2)
        q3=np.zeros(2)
        for i in range (len(q2_set)):
            wx_reached=a2*np.cos(q2_set[i]) + a3*np.cos(q2_set[i]+q3_set[i])
            wy_reached=a2*np.sin(q2_set[i]) + a3*np.sin(q2_set[i]+q3_set[i])
            if ((np.abs(wx_reached-wx)<0.001) and (np.abs(wy_reached-wy)<0.001)):
                q2[k]=q2_set[i]
                q3[k]=q3_set[i]
                k=k+1
        #Calculate the angle phi     

        phi = np.arctan2((y-wy),(x-wx))
        q4 = np.zeros(2)
        #To print the angle, uncomment the following line: 
        #Computation of q3
        q4[0] = phi-(q3[0]+q2[0])
        q4[1] = phi-(q3[1]+q2[1])

        result = []
        result.append(q1[0])
        result.append(q2[0])
        result.append(q3[0])
        result.append(q4[0])
        result.append(q1[1])
        result.append(q2[1])
        result.append(q3[1])
        result.append(q4[1])
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

        data0 = direct_kinematics_position (a1,a2,a3,a4)
        print ( 'Scorbot end-effector (x,y,z) position for angles: theta1: ' + str(a1) + ' theta2: ' + str(a2) + ' theta3: ' + str(a3) + ' theta4: ' + str(a4))
        print (data0)
        data0_ = inverse_kinematics_position (data0[0],data0[1],data0[2])
        print ( 'Scorbot end-effector angles for position: x: ' + str(data0[0]) + ' y: ' + str(data0[1]) + ' z: ' + str(data0[2]) )
        print (data0_)
        print ('     Check inverse kinematic using direct kinematic: ')
        data0__ = direct_kinematics_position (data0_[0], data0_[1], data0_[2], data0_[3])
        print ('     Solution1: ' + ' x: ' + str(data0__[0])  + ' y: ' + str(data0__[1])  + ' z: ' + str(data0__[2]))
        data0__ = direct_kinematics_position (data0_[4], data0_[5], data0_[6], data0_[7])
        print ('     Solution2: ' + ' x: ' + str(data0__[0])  + ' y: ' + str(data0__[1])  + ' z: ' + str(data0__[2]))



        #data0 = direct_kinematics_orientation (a1,a2,a3,a4)
        #print ( 'Scorbot end-effector (x,y,z) orientation for angles: theta1: ' + str(a1) + ' theta2: ' + str(a2) + ' theta3: ' + str(a3) + ' theta4: ' + str(a4))
        #print (data0)
        print ()

        data1 = direct_kinematics_position (b1, b2, b3, b4)
        print ( 'Scorbot end-effector (x,y,z) position for angles: theta1: ' + str(b1) + ' theta2: ' + str(b2) + ' theta3: ' + str(b3) + ' theta4: ' + str(b4) )
        print (data1)
        data1_ = inverse_kinematics_position (data1[0],data1[1],data1[2])
        print ( 'Scorbot end-effector angles for position: x: ' + str(data1[0]) + ' y: ' + str(data1[1]) + ' z: ' + str(data1[2]) )
        print (data1_)
        print ('     Check inverse kinematic using direct kinematic: ')
        data1__ = direct_kinematics_position (data1_[0], data1_[1], data1_[2], data1_[3])
        print ('     Solution1: ' + ' x: ' + str(data1__[0])  + ' y: ' + str(data1__[1])  + ' z: ' + str(data1__[2]))
        data1__ = direct_kinematics_position (data1_[4], data1_[5], data1_[6], data1_[7])
        print ('     Solution2: ' + ' x: ' + str(data1__[0])  + ' y: ' + str(data1__[1])  + ' z: ' + str(data1__[2]))

        #data1 = direct_kinematics_orientation (b1, b2, b3, b4)
        #print ( 'Scorbot end-effector (x,y,z) orientation for angles: theta1: ' + str(b1) + ' theta2: ' + str(b2) + ' theta3: ' + str(b3) + ' theta4: ' + str(b4))
        #print (data1)
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

