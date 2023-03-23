import random
import math
import os
import os.path
import csv
import numpy as np


directory = "models/"
model_name_prefix="scorbot_MLP_model_"
pitch_model_name_prefix="scorbot-p_MLP_model_"
training_data_prefix="scorbot_MLP_training-data-set_"
my_round = 6
First_model=0
Last_model=95

# Scorbot ER VII parameters:
# Distance are expressed in meters.
d1 = 0.358    # Distance from base center (0,0,0) rotation (1) to shoulder/body center
d2 = -0.075   # ?? Distance from center of the base to center of the shoulder/body axis
a1 = 0.050    # Distance from shoulder/body center to shoulder/body joint (2)
a2 = 0.300    # Distance from shoulder/body joint to elbow/arm joint (3)
a3 = 0.250    # Distance from elbow/arm joint to pitch/forearm joint (4)
a4 = 0.212    # End efector (gripper) length 



# End effector direct kinematic equations:
def Xe (t1,t2,t3,t4):                 # return the X for a given 4 joint angles
    return a4*math.cos(t1)*math.cos(t2+t3+t4) + a3*math.cos(t1)*math.cos(t2+t3) + a2*math.cos(t1)*math.cos(t2) + a1*math.cos(t1) + d2*math.sin(t1)
def Ye (t1,t2,t3,t4):                 # return the Y for a given 4 joint angles
    return a4*math.cos(t2+t3+t4)*math.sin(t1) + a3*math.sin(t1)*math.cos(t2+t3) + a2*math.sin(t1)*math.cos(t2) + a1*math.sin(t1) + d2*math.cos(t1)
def Ze (t1,t2,t3,t4):                 # return the Z for a given 4 joint angles
    return -a4*math.sin(t2+t3+t4) - a3*math.sin(t2+t3) - a2*math.sin(t2) + d1
def Op (t2,t3,t4):                  # Return Pitch orientation. Yaw is equal to t1 and there is no roll
    return t2+t3+t4
def Oy (t1):
    return t1

def Euclidean_distance_ (point1, point2):
    return math.sqrt ( math.pow(point1[0]-point2[0],2) + math.pow(point1[1]-point2[1],2) + math.pow(point1[2]-point2[2],2) )

def Euclidean_distance (x1,y1,z1, x2,y2,z2):
    return math.sqrt ( math.pow(x1-x2,2) + math.pow(y1-y2,2) + math.pow(z1-z2,2) )

def data_set_creation(samples, directory, training_data_file):
    Q1 = []
    Q2 = []
    Q3 = []
    Q4 = []
    posX = []
    posY = []
    posZ = []
    O_p = []
    O_y = []
    dataMat = []    
    deg2rad = math.pi/180.0
    if os.path.isfile(directory+training_data_file):
        # The training data was created before:
        #file = open (training_data_file,"r") 
        with open(directory+training_data_file, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in reader:
                Q1.append(float(row[0]))
                Q2.append(float(row[1]))
                Q3.append(float(row[2]))
                Q4.append(float(row[3]))
                posX.append(float(row[4]))
                posY.append(float(row[5]))
                posZ.append(float(row[6]))
                O_p.append(float(row[7]))
                O_y.append(float(row[8]))
                line_count += 1
            samples = line_count
        dataMat = np.c_[Q1,Q2,Q3,Q4,posX,posY,posZ,O_p,O_y] 
    else:
       # There is no training data. The training data must be created.
       if not os.path.isdir(directory):
           os.mkdir(directory)
           
       file = open (directory + training_data_file,"w")            

       # Data Set Creation
       for i in range (0,samples):
            q1= round(random.uniform ((-250.0/2.0)*deg2rad, (250.0/2.0)*deg2rad), my_round)     # Base
            q2= round(random.uniform ((-170.0/2.0)*deg2rad, (170.0/2.0)*deg2rad), my_round)     # Shoulder 
            q3= round(random.uniform ((-225.0/2.0)*deg2rad, (225.0/2.0)*deg2rad), my_round)     # Elbow    
            q4= round(random.uniform ((-180.0/2.0)*deg2rad, (180.0/2.0)*deg2rad), my_round)     # Wrist    

            Q1.append(q1)
            file.write(str(q1))
            file.write(",")

            Q2.append(q2)
            file.write(str(q2))
            file.write(",")

            Q3.append(q3)
            file.write(str(q3))
            file.write(",")

            Q4.append(q4)
            file.write(str(q4))
            file.write(",")

            X = Xe(q1,q2,q3,q4)
            posX.append(X)
            file.write(str(round(X, my_round)))
            file.write(",")

            Y = Ye(q1,q2,q3,q4)
            posY.append(Y)
            file.write(str(round(Y, my_round)))
            file.write(",")
        
            Z = Ze(q1,q2,q3,q4)
            posZ.append(Z)
            file.write(str(round(Z, my_round)))
            file.write(",")
            
            pith_orientation = Op(q2,q3,q4)
            O_p.append(pith_orientation)
            file.write(str(round(pith_orientation, my_round)))
            file.write(",")

            yaw_orientation = Oy(q1)
            O_y.append(yaw_orientation)
            file.write(str(round(yaw_orientation, my_round)))
            file.write(",")

            file.write("\n")

       file.close()
       dataMat = np.c_[Q1,Q2,Q3,Q4,posX,posY,posZ,O_p,O_y]          # Augmenting to the data matrix

    return samples, dataMat

