import random
import math
import os
import os.path
import csv
import numpy as np
from matplotlib import pyplot as plt
#import matplotlib
from mpl_toolkits.mplot3d import Axes3D
#matplotlib.use("Agg")
#from matplotlib import patches as mpatches
#from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
import keras_tuner as kt
from keras_tuner import RandomSearch
from keras_tuner import Hyperband
from keras_tuner import BayesianOptimization
import statistics
import seaborn as sb



directory = "tune_models/"
dataMat_data = []
global_data = "training-test_data.csv"
model_name = "scorbot_inverse_kinematics_MLP.h5"

results_file = "results.txt"
samples = 6000
my_round = 6

EPOCHS = 100
BS = int(0.7*samples)
EARLY_STOPPING_PATIENCE = 5
TURNER_SELECTED="bayesian"
MAX_TRIALS=500
#METRICS='mae'
METRICS='mse'

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


def Euclidean_distance (x1,y1,z1, x2,y2,z2):
    return math.sqrt ( math.pow(x1-x2,2) + math.pow(y1-y2,2) + math.pow(z1-z2,2) )


def build_model(hp):              # NN Model
    inputX = keras.Input(shape=(1,), name="Xcoor")
    inputY = keras.Input(shape=(1,), name="Ycoor")
    inputZ = keras.Input(shape=(1,), name="Zcoor")
    inputOp = keras.Input(shape=(1,), name="Opitch")
    inputOy = keras.Input(shape=(1,), name="Oyaw")
   
    my_min=16
    my_max=384
    my_step=32
    my_default=256
    
    my_activation_base1=hp.Choice('dense_activation_base1', values=['relu', 'tanh', 'sigmoid'], default='relu')
    #my_activation_base2=hp.Choice('dense_activation_base2', values=['relu', 'tanh', 'sigmoid'], default='relu')
    my_activation_shoulder1=hp.Choice('dense_activation_shoulder1', values=['relu', 'tanh', 'sigmoid', 'softmax'], default='relu')
    #my_activation_shoulder2=hp.Choice('dense_activation_shoulder2', values=['relu', 'tanh', 'sigmoid', 'softmax'], default='relu')
    my_activation_elbow1=hp.Choice('dense_activation_elbow1', values=['relu', 'tanh', 'sigmoid', 'softmax'], default='relu')
    #my_activation_elbow2=hp.Choice('dense_activation_elbow2', values=['relu', 'tanh', 'sigmoid', 'softmax'], default='relu')
    my_activation_pitch1=hp.Choice('dense_activation_pitch1', values=['relu', 'tanh', 'sigmoid', 'softmax'], default='relu')
    #my_activation_pitch2=hp.Choice('dense_activation_pitch2', values=['relu', 'tanh', 'sigmoid', 'softmax'], default='relu')
    
    # Base/q1 subnet:
    base_layer = layers.Dense(units=hp.Int('num_of_neurons_base1',min_value=my_min,max_value=my_max,step=my_step,default=my_default), activation=my_activation_base1, name="base_layer1") (inputOy)
    #base_layer = layers.Dense(units=hp.Int('num_of_neurons_base2',min_value=my_min,max_value=my_max,step=my_step,default=my_default), activation=my_activation_base2, name="base_layer2") (base_layer)
    base_q1 = layers.Dense(1, activation = "linear", name="q1")(base_layer)
    
    #Shoulder/q2 subnet:
    shoulder_input = layers.concatenate([inputX, inputZ, inputOp], name="shoulder_input")
    shoulder_layer = layers.Dense(units=hp.Int('num_of_neurons_shoulder1',min_value=my_min,max_value=my_max,step=my_step,default=my_default), activation=my_activation_shoulder1, name="shoulder_layer1") (shoulder_input)
    #shoulder_layer = layers.Dense(units=hp.Int('num_of_neurons_shoulder2',min_value=my_min,max_value=my_max,step=my_step,default=my_default), activation=my_activation_shoulder2, name="shoulder_layer2") (shoulder_layer)
    shoulder_q2 = layers.Dense(1, activation = "linear", name="q2")(shoulder_layer)
   
    #Elbow/q3 subnet:
    elbow_input = layers.concatenate([inputX, inputZ, inputOp, shoulder_q2], name="elbow_input")    
    elbow_layer = layers.Dense(units=hp.Int('num_of_neurons_elbow1',min_value=my_min,max_value=my_max,step=my_step,default=my_default), activation=my_activation_elbow1, name="elbow_layer1") (elbow_input)
    #elbow_layer = layers.Dense(units=hp.Int('num_of_neurons_elbow2',min_value=my_min,max_value=my_max,step=my_step,default=my_default), activation=my_activation_elbow2, name="elbow_layer2") (elbow_layer)
    elbow_q3 = layers.Dense(1, activation = "linear", name="q3")(elbow_layer)

    #Pitch/q4 subnet:
    pitch_input = layers.concatenate([inputX, inputZ, inputOp, elbow_q3], name="pitch_input")
    pitch_layer = layers.Dense(units=hp.Int('num_of_neurons_pitch1',min_value=my_min,max_value=my_max,step=my_step,default=my_default), activation=my_activation_pitch1, name="pitch_layer1") (pitch_input)
    #pitch_layer = layers.Dense(units=hp.Int('num_of_neurons_pitch2',min_value=my_min,max_value=my_max,step=my_step,default=my_default), activation=my_activation_pitch2, name="pitch_layer2") (pitch_layer)
    pitch_q4 = layers.Dense(1, activation = "linear", name="q4")(pitch_layer)

    model = keras.Model (inputs=[inputX, inputY, inputZ, inputOp, inputOy], outputs=[base_q1, shoulder_q2, elbow_q3, pitch_q4], name="ScorbotNN")
    
    model.summary()
    keras.utils.plot_model(model, directory+"scorbot_model_with_shape_info.png", show_shapes=True)
    
    #my_optimizer = tf.keras.optimizers.Adam(0.001)
    #my_optimizer = tf.keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4]))
    my_optimizer = tf.keras.optimizers.Adam(hp.Float("learning_rate", min_value=1e-4, max_value=1e-2))
    #optimizer = tf.keras.optimizers.SGD()
    #model.compile(optimizer=my_optimizer,loss=keras.losses.mean_squared_error, metrics=['mse'])
    #model.compile(optimizer=my_optimizer,loss=keras.losses.mean_absolute_error, metrics=['mae'])
    #model.compile(optimizer=my_optimizer,loss=keras.losses.mean_absolute_error, metrics=[METRICS])
    model.compile(optimizer=my_optimizer,loss=keras.losses.mean_squared_error, metrics=[METRICS])
    #model.compile(optimizer=my_optimizer,loss=keras.losses.mean_squared_logarithmic_error, metrics=[METRICS])

    #model.compile(optimizer=optimizer,loss=keras.losses.mean_squared_error, metrics=['mse'])
    #model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['mse'])
    #model.compile(optimizer="rmsprop", loss='mean_squared_error',metrics=['mae'])
    return model


def data_set_creation(samples, training_data_file):
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
    deg2rad = math.pi/180
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
            q2= round(random.uniform ((-170.0/2.0)*deg2rad, (170.0/2.0)*deg2rad), my_round)     # Shoulder --> Starts at 0º
            q3= round(random.uniform ((-225.0/2.0)*deg2rad, (225.0/2.0)*deg2rad), my_round)     # Elbow    --> Starts at 0º
            q4= round(random.uniform ((-180.0/2.0)*deg2rad, (180.0/2.0)*deg2rad), my_round)     # Wrist    --> Starts at 0º

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
       dataMat = np.c_[Q1,Q2,Q3,Q4,posX,posY,posZ,O_p,O_y]          # Augmenting to the data marix
       #samples, dataMat = remove_duplicates (samples, dataMat)
        #samples, dataMat = remove_duplicates (samples)

    return samples, dataMat


def save_plot(H, path):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="test_loss")
    #plt.plot(H.history["accuracy"], label="train_acc")
    #plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(path)


def title_and_labels(ax, title):
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("$X$", fontsize=16)
    ax.set_ylabel("$Y$", fontsize=16)
    ax.set_zlabel("$Z$", fontsize=16)


def filter_points_above_zero (x, y, z, error):
    x_ = []
    y_ = []
    z_ = []
    error_=[]
    for i in range(len(x)):
        if x[i] > 0 and y[i] > 0 and z[i] >0:
            x_.append(x[i])
            y_.append(y[i])
            z_.append(z[i])
            error_.append(error[i])
    return x_, y_, z_, error_


def validate_and_plot_results (dataMat,model,model_name):
    if not os.path.isfile(directory+model_name) or not model:
        print('Model '+model_name + ' does not exist. No plotting process is possible ...')
        return

    #scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    #scaler2 = preprocessing.MinMaxScaler(feature_range=(-1,1))
    data = dataMat[:,[4,5,6,7,8]]        # X,Y,Z,O_p,O_y
    output = dataMat[:,[0,1,2,3]]        # Q1,Q2,Q3,Q4
    #validate_input = data[int(0.85*samples):int(samples),:]
    #validate_output = output[int(0.85*samples):int(samples),:]
    #data_input = scaler.fit_transform(validate_input)
    #data_output = scaler2.fit_transform(validate_output)
    data_input = data
    validate_input = data
    validate_output = output
    
    #dataX_input = validate_input
    #dataX_output = validate_output
    x = data_input[:,[0]] 
    y = data_input[:,[1]] 
    z = data_input[:,[2]] 
    op = data_input[:,[3]] 
    oy = data_input[:,[4]] 
    #test_prediction = model.predict(data_input) #predict
    #q1_p,q2_p,q3_p,q4_p = model.predict(x,y,z) #predict
    q1_p,q2_p,q3_p,q4_p = model.predict([x,y,z,op,oy]) #predict
    #test_prediction = np.c_[q1_p,q2_p,q3_p,q4_p]
    #real_prediction = scaler2.inverse_transform(test_prediction)
    real_prediction = np.c_[q1_p,q2_p,q3_p,q4_p]

    # Calculate error mean and standard deviation:
    # x,y,x position:
    x = validate_input[:,[0]] 
    y = validate_input[:,[1]] 
    z = validate_input[:,[2]] 
    # Real angle data:
    #q1_r = validate_output[:,0]
    #q2_r = validate_output[:,1]
    #q3_r = validate_output[:,2]
    #q4_r = validate_output[:,3]
    # Predicted angle data:
    q1_p = real_prediction[:,0]
    q2_p = real_prediction[:,1]
    q3_p = real_prediction[:,2]
    q4_p = real_prediction[:,3]
    error = []
    for i in range(len(x)):
        t_q1 = q1_p[i]
        t_q2 = q2_p[i]
        t_q3 = q3_p[i]
        t_q4 = q4_p[i]
        x_p = Xe (t_q1, t_q2, t_q3, t_q4)
        y_p = Ye (t_q1, t_q2, t_q3, t_q4)
        z_p = Ze (t_q1, t_q2, t_q3, t_q4)
        t_x = x[i]
        t_x = t_x[0]
        t_y = y[i]
        t_y = t_y[0]
        t_z = z[i]
        t_z = t_z[0]
        error.append( Euclidean_distance(x_p,y_p,z_p,t_x,t_y,t_z) )

    res_pstdev = statistics.pstdev(error)
    res_mean = statistics.mean(error)
    print(model_name + ' mean: ' + str(res_mean))
    print(model_name + ' std dev: ' + str(res_pstdev))
    
    if os.path.isfile(directory+results_file):
        file = open (directory + results_file,"a")           
    else:
        file = open (directory + results_file,"w")   

    file.write (model_name + ' mean: ' + str(res_mean))
    file.write ("\n")
    file.write (model_name + ' std dev: ' + str(res_pstdev))
    file.write ("\n\n")
    
    # Plot angle prediction:
    plt.clf()
    plt.scatter(validate_output[:,0],real_prediction[:,0],c='b')              # Plotting Actual angles( x: desired output(Joint angles used to genarate Xe,Ye nd Titae,y: output from prediction )
    plt.scatter(validate_output[:,1],real_prediction[:,1],c='g')
    plt.scatter(validate_output[:,2],real_prediction[:,2],c='r')
    plt.scatter(validate_output[:,3],real_prediction[:,3],c='y')
    plt.xlabel('True Values angles in rad')
    plt.ylabel('Predictions  angles in rad')
    plt.title("True Value Vs Prediction")
    file_name = 'True_Value_Vs_Prediction_' + model_name + '.png'
    plt.savefig(directory+file_name)

    # Plot angle prediction (hexagonal binning):
    #fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    #fig.tight_layout()
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.5)
    fig.suptitle("Angle True Value vs Prediction",fontsize=14)
    xlim = -2.5,2.5
    ylim = -2.5,2.5
    ax[0,0].set_title('Base')
    ax[0,0].set(xlim=xlim, ylim=ylim)
    #g=sb.kdeplot(validate_output[:,0],real_prediction[:,0],color='b',ax=ax[0,0])
    ax[0,0].hexbin(validate_output[:,0],real_prediction[:,0],cmap='Blues',gridsize=25)
    ax[0,0].set_xlabel('True Values angles in rad')
    ax[0,0].set_ylabel('Predictions angles in rad')
    ax[0,1].set_title('Shoulder')
    ax[0,1].set(xlim=xlim, ylim=ylim)
    ax[0,1].hexbin(validate_output[:,1],real_prediction[:,1],cmap='Greens',gridsize=25)
    ax[0,1].set( xlabel = "True Values angles in rad", ylabel = "Predictions angles in rad")    
    ax[1,0].set_title('Elbow')
    ax[1,0].set(xlim=xlim, ylim=ylim)
    ax[1,0].hexbin(validate_output[:,2],real_prediction[:,2],cmap='Reds',gridsize=25)
    ax[1,0].set( xlabel = "True Values angles in rad", ylabel = "Predictions angles in rad")    
    ax[1,1].set(xlim=xlim, ylim=ylim)
    ax[1,1].set_title('Wrist')
    ax[1,1].hexbin(validate_output[:,3],real_prediction[:,3],cmap='Purples',gridsize=25)
    ax[1,1].set( xlabel = "True Values angles in rad", ylabel = "Predictions angles in rad")    
    file_name = 'True_Value_Vs_Prediction_hexbin_' + model_name + '.png'
    plt.savefig(directory+file_name)

    # Plot error distribution:
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sb.distplot(error, hist=False, color='lightblue', kde=True, kde_kws={'shade': True, 'linewidth': 2})
    plt.axvline(x=res_mean, color='r')
    plt.annotate ('Mean', color='r', size=14, xytext=(res_mean+0.1,1.3), xy=(res_mean+0.01,1.0), arrowprops=dict(facecolor='r',shrink=0.01))
    ax.set_title("Error distribution", fontsize=20)
    ax.set_xlabel("Error (metres)", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    plt.savefig(directory+"error_histogram_sb.png")

    # Density Plot and Histogram of errors
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sb.distplot(error, hist=True, kde=True, bins = int(samples/20), color = 'blue', hist_kws={'edgecolor':'lightblue'}, kde_kws={'linewidth': 3})
    plt.axvline(x=res_mean, color='r')
    plt.annotate ('Mean', color='r', size=14, xytext=(res_mean+0.1,2.1), xy=(res_mean+0.01,1.9), arrowprops=dict(facecolor='r',shrink=0.01))
    ax.set_title("Error distribution", fontsize=20)
    ax.set_xlabel("Error (metres)", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    plt.savefig(directory+"error_histogram_and_density_sb.png")

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.hist(error, color='lightblue', bins = int(samples/20))
    plt.axvline(x=res_mean, color='r')
    plt.annotate ('Mean', color='r', size=14, xytext=(res_mean+0.1,120), xy=(res_mean+0.01,100), arrowprops=dict(facecolor='r',shrink=0.01))
    ax.set_title("Error distribution", fontsize=20)
    ax.set_xlabel("Error (metres)", fontsize=16)
    ax.set_ylabel("Number of points", fontsize=16)
    plt.savefig(directory+"error_histogram_hist.png")
 
    # 3D plots:
    # Error plot 1:
    fig = plt.figure(figsize=(18,12))
    ax = plt.axes (projection='3d')
    p=ax.scatter(x,y,z, s=len(x), c=error, cmap='Blues') #, vmin=vmin, vmax=vmax)
    fig.colorbar(p, ax=ax, shrink=0.5)
    ax.set(facecolor='gainsboro')
    ax.plot([0, 0.9], [0, 0], [0, 0], color='red') # Start_x, End_x, Start_y, End_y, Start_z, End_z
    ax.plot([0, 0], [0, 0.9], [0, 0], color='orange') # Start_x, End_x, Start_y, End_y, Start_z, End_z
    ax.plot([0, 0], [0, 0], [0, 1.1], color='peru') # Start_x, End_x, Start_y, End_y, Start_z, End_z
    ax.plot([0,-0.5], [0, 0], [0, 0], color='red') # Start_x, End_x, Start_y, End_y, Start_z, End_z
    ax.plot([0, 0], [0, -0.8], [0, 0], color='orange') # Start_x, End_x, Start_y, End_y, Start_z, End_z
    ax.plot([0, 0], [0, 0], [0, -0.5], color='peru') # Start_x, End_x, Start_y, End_y, Start_z, End_z
    title_and_labels(ax, "Error magnitudes in Scorbot-ER VII Working Area")
    plt.legend()
    plt.savefig(directory+"error1.png")
    # Error plot 2:
    fig = plt.figure(figsize=(18,12))
    ax = plt.axes (projection='3d')
    x_ = []
    y_ = []
    z_ = []
    error_=[]
    x_, y_, z_, error_ = filter_points_above_zero (x, y, z, error)
    p=ax.scatter(x_,y_,z_, s=len(x_), c=error_, cmap='Blues')
    fig.colorbar(p, ax=ax, shrink=0.6)
    ax.set(facecolor='lightgrey')
    ax.plot([0, 0.9], [0, 0], [0, 0], color='red') # Start_x, End_x, Start_y, End_y, Start_z, End_z
    ax.plot([0, 0], [0, 0.9], [0, 0], color='red') # Start_x, End_x, Start_y, End_y, Start_z, End_z
    ax.plot([0, 0], [0, 0], [0, 1.1], color='red') # Start_x, End_x, Start_y, End_y, Start_z, End_z
    title_and_labels(ax, "Error magnitudes in Scorbot-ER VII Working Area (x>0,y>0,z>0)")
    plt.legend()
    plt.savefig(directory+"error2.png")


def tune_and_save_best_model(dataMat,model_name):
    if os.path.isfile(directory+model_name):
        print('Model '+model_name + ' already exists. Loaded. No tuning process is performed ...')
        model = tf.keras.models.load_model(directory + model_name)
        return model
    
    # Model must be trained:
    data = dataMat[:,[4,5,6,7,8]]        # X,Y,Z,O_p,O_y
    output = dataMat[:,[0,1,2,3]]    # Q1,Q2,Q3,Q4

    train_input = data[0:int(0.7*samples),:]                             #Separate data set in to Train, Test And Validation
    train_output = output[0:int(0.7*samples),:]
    validation_input= data[int(0.7*samples):samples,:]
    validation_output= output[int(0.7*samples):samples,:]
    
    # We don't Scale down all the features to a same range.
    dataI = train_input
    dataO = train_output
    dataI_validation = validation_input
    dataO_validation = validation_output
    
    x = dataI[:,[0]] 
    y = dataI[:,[1]] 
    z = dataI[:,[2]] 
    op = dataI[:,[3]] 
    oy = dataI[:,[4]] 
    q1_ = dataO[:,[0]] 
    q2_ = dataO[:,[1]] 
    q3_ = dataO[:,[2]] 
    q4_ = dataO[:,[3]] 

    x_v = dataI_validation[:,[0]] 
    y_v = dataI_validation[:,[1]] 
    z_v = dataI_validation[:,[2]] 
    op_v = dataI_validation[:,[3]] 
    oy_v = dataI_validation[:,[4]] 
    q1_v = dataO_validation[:,[0]] 
    q2_v = dataO_validation[:,[1]] 
    q3_v = dataO_validation[:,[2]] 
    q4_v = dataO_validation[:,[3]] 
    
    if TURNER_SELECTED=="hyperband":
        # instantiate the hyperband tuner object
        print("[INFO] instantiating a hyperband tuner object...")
        tuner = Hyperband(build_model, objective="mean_absolute_error", max_epochs=EPOCHS, factor=3, seed=42, directory=directory, project_name="hyperband")
    elif TURNER_SELECTED=="random":
        # instantiate the random search tuner object
        print("[INFO] instantiating a random search tuner object...")        
        tuner = RandomSearch(build_model, objective="mean_absolute_error", max_trials=MAX_TRIALS, seed=42, directory=directory, project_name="random")
    else:
        # instantiate the bayesian optimization tuner object
        print("[INFO] instantiating a bayesian optimization tuner object...")
        tuner = BayesianOptimization(build_model, objective=kt.Objective('val_loss',direction='min'), max_trials=MAX_TRIALS, seed=42, directory=directory, project_name="bayesian")
        
        
    # perform the hyperparameter search
    print("[INFO] performing hyperparameter search...")

    # initialize an early stopping callback to prevent the model from
    # overfitting/spending too much time training with minimal gains
    NAME = "Scorbot NN Hyperparameters"
    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))           # Create    callbacks for tensorboard visualizations

    es = EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)

    tuner.search( {'Xcoor':x,'Ycoor':y,'Zcoor':z, 'Opitch':op, 'Oyaw':oy}, {'q1':q1_,'q2':q2_,'q3':q3_,'q4':q4_}, validation_data=([x_v,y_v,z_v,op_v,oy_v], [q1_v,q2_v,q3_v,q4_v]), batch_size=BS, callbacks=[tensorboard, es], epochs=EPOCHS )

    # Show a summary of the search
    print("\n")
    print("Tuner summary: ")
    tuner.search_space_summary()
    print("\n")
   
    # grab the best hyperparameters
    bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
    print ("TURNER_SELECTED: "+TURNER_SELECTED)
    print ("Layers:")
    print ("-------")
    print("[INFO] optimal number of filters in num_of_neurons_base1 layer: {}".format(bestHP.get("num_of_neurons_base1")))
    print("[INFO] optimal number of filters in num_of_neurons_base2 layer: {}".format(bestHP.get("num_of_neurons_base2")))
    print("[INFO] optimal number of filters in num_of_neurons_shoulder1 layer: {}".format(bestHP.get("num_of_neurons_shoulder1")))
    print("[INFO] optimal number of filters in num_of_neurons_shoulder2 layer: {}".format(bestHP.get("num_of_neurons_shoulder2")))
    print("[INFO] optimal number of filters in num_of_neurons_elbow1 layer: {}".format(bestHP.get("num_of_neurons_elbow1")))
    print("[INFO] optimal number of filters in num_of_neurons_elbow2 layer: {}".format(bestHP.get("num_of_neurons_elbow2")))
    print("[INFO] optimal number of filters in num_of_neurons_pitch1 layer: {}".format(bestHP.get("num_of_neurons_pitch1")))
    print("[INFO] optimal number of filters in num_of_neurons_pitch2 layer: {}".format(bestHP.get("num_of_neurons_pitch2")))
    
    print ("Learning rate:")
    print ("--------------")
    print("[INFO] optimal learning rate: {:.4f}".format(bestHP.get("learning_rate")))
 
    print ("Activation function:")
    print ("-------")
    print("[INFO] optimal dense_activation_base1: {}".format(bestHP.get("dense_activation_base1")))
    print("[INFO] optimal dense_activation_base2: {}".format(bestHP.get("dense_activation_base2")))
    print("[INFO] optimal dense_activation_shoulder1: {}".format(bestHP.get("dense_activation_shoulder1")))
    print("[INFO] optimal dense_activation_shoulder2: {}".format(bestHP.get("dense_activation_shoulder2")))
    print("[INFO] optimal dense_activation_elbow1: {}".format(bestHP.get("dense_activation_elbow1")))
    print("[INFO] optimal dense_activation_elbow2: {}".format(bestHP.get("dense_activation_elbow2")))
    print("[INFO] optimal dense_activation_pitch1: {}".format(bestHP.get("dense_activation_pitch1")))
    print("[INFO] optimal dense_activation_pitch2: {}".format(bestHP.get("dense_activation_pitch2")))
    tuner.results_summary()
    
    
    # build the best model and train it
    print("[INFO] training the best model...")
    model = tuner.hypermodel.build(bestHP)
    history = model.fit({'Xcoor':x,'Ycoor':y,'Zcoor':z, 'Opitch':op, 'Oyaw':oy}, {'q1':q1_,'q2':q2_,'q3':q3_,'q4':q4_}, validation_data=([x_v,y_v,z_v,op_v,oy_v], [q1_v,q2_v,q3_v,q4_v]), batch_size=BS, epochs=EPOCHS, callbacks=[es], verbose=0)

    # Save best model:
    model.save (directory+ model_name)

    # plot loss during training
    plt.figure()
    plt.title('Loss '+"(using "+TURNER_SELECTED+" tunning)")
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    plt.savefig(directory+'Loss_and_' +METRICS + '_(' +TURNER_SELECTED+ ')' +  '.png')
    
    return model
    


if __name__ == '__main__':

    samples, dataMat = data_set_creation(samples, global_data)
    model = tune_and_save_best_model(dataMat,model_name)
    validate_and_plot_results (dataMat,model,model_name)

