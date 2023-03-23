from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sb
import statistics

import math
import os
import os.path
import csv
import numpy as np

########### 2D plots:

def angles_scatter_plot (dataMat,model,directory,model_name_prefix):
    # Plot angle prediction:
    output = dataMat[:,[0,1,2,3]]        # Q1,Q2,Q3,Q4
    data_input = dataMat[:,[4,5,6,7,8]]        # X,Y,Z,O_p,O_y
    x = data_input[:,[0]] 
    y = data_input[:,[1]] 
    z = data_input[:,[2]] 
    op = data_input[:,[3]] 
    oy = data_input[:,[4]] 
    q1_p,q2_p,q3_p,q4_p = model.predict([x,y,z,op,oy]) #predict
    prediction = np.c_[q1_p,q2_p,q3_p,q4_p]

    plt.clf()
    # Plotting Actual angles
    plt.scatter(output[:,0],prediction[:,0],c='b')              
    plt.scatter(output[:,1],prediction[:,1],c='g')
    plt.scatter(output[:,2],prediction[:,2],c='r')
    plt.scatter(output[:,3],prediction[:,3],c='purple')
    plt.xlabel('True Values angles in rad')
    plt.ylabel('Predictions  angles in rad')
    plt.title("True Value Vs Prediction")
    file_name = 'True_Value_Vs_Prediction_' + model_name_prefix + '.png'
    plt.savefig(directory+file_name)
    

def angles_hexbin_plot (dataMat,model,directory,model_name_prefix):
    # Plot angle prediction (hexagonal binning):
    output = dataMat[:,[0,1,2,3]]        # Q1,Q2,Q3,Q4
    data_input = dataMat[:,[4,5,6,7,8]]        # X,Y,Z,O_p,O_y
    x = data_input[:,[0]] 
    y = data_input[:,[1]] 
    z = data_input[:,[2]] 
    op = data_input[:,[3]] 
    oy = data_input[:,[4]] 
    q1_p,q2_p,q3_p,q4_p = model.predict([x,y,z,op,oy]) #predict
    prediction = np.c_[q1_p,q2_p,q3_p,q4_p]

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    #fig.tight_layout()
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.5)
    fig.suptitle("Angle True Value vs Prediction",fontsize=16)
    xlim = -2.5,2.5
    ylim = -2.5,2.5
    ax[0,0].set_title('Base', fontsize=15)
    ax[0,0].tick_params(labelsize=14)
    ax[0,0].set(xlim=xlim, ylim=ylim)
    ax[0,0].hexbin(output[:,0],prediction[:,0],cmap='Blues',gridsize=25)
    #ax[0,0].set( xlabel = "True Values angles in rad", ylabel = "Predictions angles in rad") 
    ax[0,0].set_xlabel("True Values angles in rad", fontsize=14)
    ax[0,0].set_ylabel("Predicted angles in rad", fontsize=14)
    
    ax[0,1].set_title('Shoulder', fontsize=15)
    ax[0,1].tick_params(labelsize=14)
    ax[0,1].set(xlim=xlim, ylim=ylim)
    ax[0,1].hexbin(output[:,1],prediction[:,1],cmap='Greens',gridsize=25)
    #ax[0,1].set( xlabel = "True Values angles in rad", ylabel = "Predictions angles in rad")  
    ax[0,1].set_xlabel("True Values angles in rad", fontsize=14)
    ax[0,1].set_ylabel("Predicted angles in rad", fontsize=14)
    
    ax[1,0].set_title('Elbow', fontsize=15)
    ax[1,0].tick_params(labelsize=14)
    ax[1,0].set(xlim=xlim, ylim=ylim)
    ax[1,0].hexbin(output[:,2],prediction[:,2],cmap='Reds',gridsize=25)
    #ax[1,0].set( xlabel = "True Values angles in rad", ylabel = "Predictions angles in rad")  
    ax[1,0].set_xlabel("True Values angles in rad", fontsize=14)
    ax[1,0].set_ylabel("Predicted angles in rad", fontsize=14)
    
    ax[1,1].set(xlim=xlim, ylim=ylim)
    ax[1,1].set_title('Wrist', fontsize=15)
    ax[1,1].tick_params(labelsize=14)
    ax[1,1].hexbin(output[:,3],prediction[:,3],cmap='Purples',gridsize=25)
    #ax[1,1].set( xlabel = "True Values angles in rad", ylabel = "Predictions angles in rad")  
    ax[1,1].set_xlabel("True Values angles in rad", fontsize=14)
    ax[1,1].set_ylabel("Predicted angles in rad", fontsize=14)
    
    file_name = 'True_Value_Vs_Prediction_hexbin_' + model_name_prefix + '.png'
    plt.savefig(directory+file_name)


def density_error_plot (samples, error, mean, directory, model_name_prefix):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sb.distplot(error, hist=False, color='lightblue', kde=True, kde_kws={'shade': True, 'linewidth': 2})
    plt.axvline(x=mean, color='r')
    plt.annotate ('Mean', color='r', size=14, xytext=(mean+0.1,6.3), xy=(mean+0.01,6.0), arrowprops=dict(facecolor='r',shrink=0.01))
    ax.set_title("Error distribution", fontsize=20)
    ax.set_xlabel("Error (metres)", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    file_name = 'Inverse_kinematics_error_density_' + model_name_prefix + '.png'
    plt.savefig(directory+file_name)


def histogram_and_density_error_plot (samples, error, mean, directory, model_name_prefix):
    #sb.set(font_scale=2)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sb.distplot(error, hist=True, kde=True, bins = int(samples/20), color = 'blue', hist_kws={'edgecolor':'lightblue'}, kde_kws={'linewidth': 3})
    #sb.set(font_scale=1)
    plt.axvline(x=mean, color='r')
    plt.annotate ('Mean', color='r', size=28, xytext=(mean+0.1,7.5), xy=(mean+0.01,6.5), arrowprops=dict(facecolor='r',shrink=0.01))
    ax.set_title("Error distribution", fontsize=30)
    ax.set_xlabel("Error (metres)", fontsize=28)
    ax.set_ylabel("Density", fontsize=28)
    ax.tick_params(labelsize=26)
    file_name = 'Inverse_kinematics_error_histogram_and_density_' + model_name_prefix + '.png'
    plt.savefig(directory+file_name)
    
    
def histogram_error_plot (samples, error, mean, directory, model_name_prefix):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.hist(error, color='lightblue', bins = int(samples/20))
    plt.axvline(x=mean, color='r')
    plt.annotate ('Mean', color='r', size=14, xytext=(mean+0.1,80), xy=(mean+0.01,65), arrowprops=dict(facecolor='r',shrink=0.01))
    ax.set_title("Error distribution", fontsize=20)
    ax.set_xlabel("Error (metres)", fontsize=16)
    ax.set_ylabel("Number of points", fontsize=16)
    file_name = 'Inverse_kinematics_error_histogram_' + model_name_prefix + '.png'
    plt.savefig(directory+file_name)
   
    
    # Error matrix plot. It compares errors (mean and std deviation) when using 
    # several models to predict inverse kinematics:    
def bar_error_matrix_plot (error_matrix, Last_model,directory):
    mean_vector=np.zeros(Last_model+1,dtype=float)
    std_dev_vector=np.zeros(Last_model+1,dtype=float)
    for i in range (0,Last_model+1):
        model_i=error_matrix[:,i]
        mean_vector[i]=statistics.mean(model_i)
        std_dev_vector[i]=statistics.pstdev(model_i)
    # Plot histograms:
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x=np.arange(1,Last_model+2,1)
    ax = fig.add_axes([0,0,1,1])
    ax.bar(x,mean_vector,color='salmon')
    ax.bar(x,std_dev_vector,color=sb.desaturate("indianred",.75))
    #ax.set_title("Bootstrap sampling models (1 to 100). Mean and standard deviation error", fontsize=20)
    ax.set_xlabel("Number of models (MLP ANNs) used", fontsize=18)
    ax.set_ylabel("Error (metres)", fontsize=18)
    ax.tick_params(labelsize=16)
    ax.legend(labels=['Mean', 'Standard deviation'], fontsize = 18)
    ax.set_yticks(np.arange(0, .13, .01))
    ax.set_xticks(np.arange(0, 101, 10))
    file_name = 'histogram_error_matrix.png'
    plt.show()
    plt.savefig(directory+file_name)
    np.savetxt(directory+'mean_error_vector.txt', mean_vector,fmt='%f')
    np.savetxt(directory+'std.dev_error_vector.txt', std_dev_vector,fmt='%f')

def bar_error_matrix_plot2 (error_matrix, Last_model,directory):
    mean_vector=np.zeros(Last_model+1,dtype=float)
    std_dev_vector=np.zeros(Last_model+1,dtype=float)
    for i in range (0,Last_model+1):
        model_i=error_matrix[:,i]
        mean_vector[i]=statistics.mean(model_i)
        std_dev_vector[i]=statistics.pstdev(model_i)
    # Plot histograms:
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x=np.arange(1,Last_model+2,1)
    ax = fig.add_axes([0,0,1,1])
    ax.bar(x,mean_vector,color='salmon')
    ax.bar(x,std_dev_vector,color=sb.desaturate("indianred",.75))
    #sb.barplot(x=x,y=mean_vector, color='salmon', saturation=.5, errwidth=5)
    #sb.barplot(x=x,y=std_dev_vector, color=sb.desaturate("indianred",.75), saturation=.5, errwidth=5)
    ax.set_title("Bootstrapping models (1 to 100) tested. Mean and standard deviation error", fontsize=18)
    ax.set_xlabel("Number of models (MLP ANNs) used", fontsize=16)
    ax.set_ylabel("Error (metres)", fontsize=16)
    ax.legend(labels=['Mean', 'Standard deviation'])
    ax.set_yticks(np.arange(0, .21, .02))
    ax.set_xticks(np.arange(0, 101, 10))
    file_name = 'histogram_error_matrix.png'
    plt.show()
    plt.savefig(directory+file_name)
    np.savetxt(directory+'mean_error_vector.txt', mean_vector,fmt='%f')
    np.savetxt(directory+'std.dev_error_vector.txt', std_dev_vector,fmt='%f')
    
########### 3D plots:
    
# Auxiliary functions for the main 3D plot functions:
    
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


# Main 3D plot functions:

def threeD_error_plot (error, dataMat, directory, model_name_prefix):
    data = dataMat[:,[4,5,6,7,8]]        # X,Y,Z,O_p,O_y
    x = data[:,[0]] 
    y = data[:,[1]] 
    z = data[:,[2]] 
    # Plot default view:
    fig = plt.figure(figsize=(18,12))
    ax = plt.axes (projection='3d')
    p=ax.scatter(x,y,z, s=len(x), c=error, cmap='Blues') #, vmin=vmin, vmax=vmax)
    fig.colorbar(p, ax=ax, shrink=0.5)
    ax.set(facecolor='gainsboro')
    # Paint axes: this is optional
    #ax.plot([0, 0.9], [0, 0], [0, 0], color='red') # Start_x, End_x, Start_y, End_y, Start_z, End_z
    #ax.plot([0, 0], [0, 0.9], [0, 0], color='orange') # Start_x, End_x, Start_y, End_y, Start_z, End_z
    #ax.plot([0, 0], [0, 0], [0, 1.1], color='peru') # Start_x, End_x, Start_y, End_y, Start_z, End_z
    #ax.plot([0,-0.5], [0, 0], [0, 0], color='red') # Start_x, End_x, Start_y, End_y, Start_z, End_z
    #ax.plot([0, 0], [0, -0.8], [0, 0], color='orange') # Start_x, End_x, Start_y, End_y, Start_z, End_z
    #ax.plot([0, 0], [0, 0], [0, -0.5], color='peru') # Start_x, End_x, Start_y, End_y, Start_z, End_z
    title_and_labels(ax, "Errors (in metres) in the Scorbot-ER VII Working Area")
    plt.legend()
    file_name = 'Inverse_kinematics_error_3D_' + model_name_prefix + '.png'
    plt.savefig(directory+file_name)
    # Plot some more views:
    # 45º up and 48º rotate:
    fig = plt.figure(figsize=(18,12))
    ax = plt.axes (projection='3d')
    ax.view_init(elev=45., azim=48)
    p=ax.scatter(x,y,z, s=len(x), c=error, cmap='Blues')
    fig.colorbar(p, ax=ax, shrink=0.6)
    ax.set(facecolor='lightgrey')
    title_and_labels(ax, "Error magnitudes in Scorbot-ER VII Working Area")
    plt.legend()
    file_name = 'Inverse_kinematics_error_3D_rotate_48_' + model_name_prefix + '.png'
    plt.savefig(directory+file_name)
    # 45º up and 72º rotate:
    fig = plt.figure(figsize=(18,12))
    ax = plt.axes (projection='3d')
    ax.view_init(elev=45., azim=72)
    p=ax.scatter(x,y,z, s=len(x), c=error, cmap='Blues')
    fig.colorbar(p, ax=ax, shrink=0.6)
    ax.set(facecolor='lightgrey')
    title_and_labels(ax, "Error magnitudes in Scorbot-ER VII Working Area")
    plt.legend()
    file_name = 'Inverse_kinematics_error_3D_rotate_72_' + model_name_prefix + '.png'
    plt.savefig(directory+file_name)
    


def threeD_error_plot_first_quadrant (error, dataMat, directory, model_name_prefix):
    data = dataMat[:,[4,5,6,7,8]]        # X,Y,Z,O_p,O_y
    x = data[:,[0]] 
    y = data[:,[1]] 
    z = data[:,[2]] 
    fig = plt.figure(figsize=(18,12))
    ax = plt.axes (projection='3d')
    ax.tick_params(labelsize=22)
    x_ = []
    y_ = []
    z_ = []
    error_=[]
    x_, y_, z_, error_ = filter_points_above_zero (x, y, z, error)
    p=ax.scatter(x_,y_,z_, s=len(x_), c=error_, cmap='Blues')
    cb = fig.colorbar(p, ax=ax, shrink=0.6)
    for t in cb.ax.get_yticklabels():
     t.set_fontsize(22)
    ax.set(facecolor='lightgrey')
    # Paint axes: this is optional
    ax.plot([0, 0.9], [0, 0], [0, 0], color='red') # Start_x, End_x, Start_y, End_y, Start_z, End_z
    ax.plot([0, 0], [0, 0.9], [0, 0], color='red') # Start_x, End_x, Start_y, End_y, Start_z, End_z
    ax.plot([0, 0], [0, 0], [0, 1.1], color='red') # Start_x, End_x, Start_y, End_y, Start_z, End_z
    title_and_labels(ax, "Error magnitudes in Scorbot-ER VII Working Area (x>0,y>0,z>0)")
    #ax.set_xlabel('$X$', fontsize=28, rotation=0)
    #ax.set_ylabel('$Y$', fontsize=28, rotation=90)
    #ax.set_zlabel('$Z$', fontsize=28, rotation=90)
    ax.set_xlabel('', fontsize=28, rotation=0)
    ax.set_ylabel('', fontsize=28, rotation=90)
    ax.set_zlabel('', fontsize=28, rotation=90)
    file_name = 'Inverse_kinematics_error_3D_1st_quadrant_' + model_name_prefix + '.png'
    plt.savefig(directory+file_name)
