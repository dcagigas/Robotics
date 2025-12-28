import random
import math
import numpy as np
from matplotlib import pyplot as plt

# README:
# This program simulates a Cartesian robot which has 3 links and can only move on planer surface.
# It has 3 joint angles (Q1,Q2,Q3) with constrains for each joint angle.
# Q1 can only go from 0 to (-180) degrees.
# Q2 can only go from (-180) to 0 degrees.
# Q3 can only go from (-90) to 90 degrees.
# The robot has 3 links: A, B and C.
# Link A starts in Q1 and ends in Q2.
# Link B starts in Q2 and ends in Q3.
# Link C starts in Q3 and ends in the robot end effector or just robot end.
# It uses de classical Jacobian. Plese see this blog:
# https://medium.com/unity3danimation/overview-of-jacobian-ik-a33939639ab2

A = 0.7  # A robot link Length
B = 0.5  # B robot link Length
C = 0.3  # C robot link Length

H = 1           # 'H' is just a simulation step that can be tuned (1, 0.1, 0.01, 0.001, etc)
EPSILON = 0.05  # Maximum error distance between target and the final position of the end effector/hand


def Euclidean_distance (x1,y1,x2,y2):
    return math.sqrt ( math.pow(x1-x2,2) + math.pow(y1-y2,2) )

# Direct kinematic equations:
def Xe (q1,q2,q3):                 # return the X,Y,Tita for a given 3 joint angles
    return A*math.cos(q1) + B*math.cos(q1 + q2) + C*math.cos(q1 + q2 + q3)
def Ye (q1,q2,q3):
    return A*math.sin(q1) + B*math.sin(q1 + q2) + C*math.sin(q1 + q2 + q3)
def tita (q1,q2,q3):
    return math.degrees(q1) + math.degrees(q2) + math.degrees(q3)

def draw_plot (X,Y,q1,q2,q3,i=0):
    x1 = A*math.cos(q1)
    y1 = A*math.sin(q1)
    x2 = B*math.cos(q1+q2)
    y2 = B*math.sin(q1+q2)
    x3 = C*math.cos(q1+q2+q3)
    y3 = C*math.sin(q1+q2+q3)
    plt.clf()
    plt.plot([0,x1],[0,y1],'-k')                        # Draw link A                
    plt.plot([x1,x1+x2],[y1,y1+y2],'-k')                # Draw link B
    plt.plot([x1+x2,x1+x2+x3],[y1+y2,y1+y2+y3],'-k')    # Draw link C
    plt.scatter(x1+x2+x3, y1+y2+y3, c='r')              # Current robot end point/Hand
    plt.scatter(X, Y, c='g')            # Target end point that robot must reach
    plt.scatter(0, 0, c='b')            # q1 robot rotation/angle
    plt.scatter(x1, y1, c='b')          # q2 robot rotation/angle
    plt.scatter(x1+x2, y1+y2, c='b')    # q3 robot rotation/angle
    error = Euclidean_distance (x1+x2+x3,y1+y2+y3,X,Y)
    plt.title("Step: " + str(i) + "   Error: " + str(round(error,3)) + "\nTarget: ({},{})".format(round (X,3),round(Y,3)) + "\nEnd-Effector/Hand position: ({},{})".format(round (x1+x2+x3,3),round(y1+y2+y3,3)) )
    plt.show()
    #print ("Target: ({},{})".format(round (X,3),round(Y,3)))

def get_random_angles ():
    q1= round(random.uniform(0,math.pi),5)              # Q1 can only go from 0 to (-180) degrees
    q2= round(random.uniform(-math.pi,0),5)             # Q2 can only go from (-180) to 0 degrees
    q3= round(random.uniform(-math.pi/2, math.pi/2), 5) # Q3 can only go from (-90) to 90 degrees
    return q1, q2, q3

def get_random_target ():
    q1= round(random.uniform(0,math.pi),5)              # Q1 can only go from 0 to (-180) degrees
    q2= round(random.uniform(-math.pi,0),5)             # Q2 can only go from (-180) to 0 degrees
    q3= round(random.uniform(-math.pi/2, math.pi/2), 5) # Q3 can only go from (-90) to 90 degrees
    x = Xe (q1,q2,q3)
    y = Ye (q1,q2,q3)
    t = tita (q1,q2,q3)
    return x, y, t

    # Functions to calculate the Inverse Kinematics (IK) using the Jacovian:
def Jacobian (angles):
    q1 = angles[0]
    q2 = angles[1]
    q3 = angles[2]
    # Partial derivates of Xe respect q1, q2 and q3:
    # Xe: A*math.cos(q1) + B*math.cos(q1 + q2) + C*math.cos(q1 + q2 + q3)
    row_x = np.array([-A*math.sin(q1) - B*math.sin(q1 + q2) - C*math.sin(q1 + q2 + q3), -B*math.sin(q1 + q2) - C*math.sin(q1 + q2 + q3), -C*math.sin(q1 + q2 + q3)])
    # Partial derivates of Ye respect q1, q2 and q3:
    # Ye: A*math.sin(q1) + B*math.sin(q2 + q3) + C*math.sin(q1 + q2 + q3)
    row_y = np.array([A*math.cos(q1) + C*math.cos(q1 + q2 + q3), B*math.cos(q2 + q3) + C*math.cos(q1 + q2 + q3), B*math.cos(q2 + q3) + C*math.cos(q1 + q2 + q3)])
    return np.array([row_x, row_y])                  


def d0 (end, target, angles):
    V = np.array(target)-np.array(end)
    J = Jacobian(angles)
    return np.matmul(V, J)


def IK_calculation (target, angles):
    angles = np.array(angles)
    q1 = angles[0]
    q2 = angles[1]
    q3 = angles[2]
    pos_x = Xe(q1,q2,q3)
    pos_y = Ye(q1,q2,q3)
    error = Euclidean_distance (target[0], target[1], pos_x, pos_y)
    steps = 0
    while (error > EPSILON and steps < 100):
        J = d0([pos_x,pos_y],target,[q1,q2,q3])*H
        angles = angles + J
        q1 = angles[0]
        q2 = angles[1]
        q3 = angles[2]
        pos_x = Xe(q1,q2,q3)
        pos_y = Ye(q1,q2,q3)
        error = Euclidean_distance (target[0], target[1], pos_x, pos_y)
        steps = steps + 1
    return angles, steps, error, pos_x, pos_y

                      
if __name__ == '__main__':
    q1,q2,q3 = get_random_angles ()
    print ("Initial angles: " + str([q1,q2,q3]))
    x_t = Xe (q1,q2,q3)
    y_t = Ye (q1,q2,q3)
    #print ("End-Effector/Hand position: ({},{})".format(round(x_t,3), round(y_t,3)) )
    x,y,t = get_random_target ()
    draw_plot (x,y,q1,q2,q3)

    angles, steps, error, pos_x, pos_y = IK_calculation ([x,y], [q1,q2,q3])
    
    draw_plot (x,y,angles[0],angles[1],angles[2],steps)
    print ("Final angles: " + str(angles))
    print ("Error: " + str(round(error,3)))
    print ("Target pos_x: "+str(round(x,3))+" Target pos_y: "+str(round(y,3)))
    print ("Hand pos_x: "+str(round(pos_x,3))+" Hand pos_y: "+str(round(pos_y,3)))
