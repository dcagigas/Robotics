from scorbot_common_functions_inverse_kinematics_hybrid import *
from scorbot_plot_functions_inverse_kinematics import *
import tensorflow as tf
import sys
#from tensorflow import keras

test_samples_size_error_matrix=100
test_data_file_name_error_matrix="test_data_set_100.csv"
error_data_file_error_matrix="models_error_matrix_100.csv"

test_samples_size=5000
test_data_file_name="test_data_set.csv"
error_data_file="models_error_matrix.csv"

error_vector_model_prefix="error_vector_model_"
number_vector_model_prefix="number_vector_model_"

result_directory="model_results/"


def load_models (directory, model_name_prefix, model_number):
    model_list=[]
    if not os.path.isdir(directory):
        print ("Error: directory " + str(directory) + " does not exist. Models (MLPs) can't be loaded")
        return model_list
    my_model_range=range(model_number)
    for i in my_model_range:
        model_name = model_name_prefix + str(i) + ".h5"
        if os.path.isfile (directory + model_name):
            model = tf.keras.models.load_model(directory + model_name)
            model_list.append(model)
        else:
            print("Warning: model " + model_name + " does not exist in directory " + directory + ". Not loaded")
    return model_list


def get_error (end_point, model):
    # end_point: [x,y,x,pitch,yaw]
    # angles: [q1,q2,q3,q4]
    x=end_point[0]
    y=end_point[1]
    z=end_point[2]
    pitch=end_point[3]
    yaw=end_point[4]
    # Inverse kinematics provided by the model:
    #q1_p,q2_p,q3_p,q4_p = model.predict([x,y,z,pitch,yaw])
    q1_p,q2_p,q3_p,q4_p = model([x,y,z,pitch,yaw], training=False)
    q1_p = q1_p[0]
    q1_p = float(q1_p[0])
    q2_p = q2_p[0]
    q2_p = float(q2_p[0])
    q3_p = q3_p[0]
    q3_p = float(q3_p[0])
    q4_p = q4_p[0]
    q4_p = float(q4_p[0])
    # Direct kinematics provided by Denavit-Hartenberg:
    x_p=Xe (q1_p,q2_p,q3_p,q4_p)
    y_p=Ye (q1_p,q2_p,q3_p,q4_p)
    z_p=Ze (q1_p,q2_p,q3_p,q4_p)
    error=Euclidean_distance(x_p,y_p,z_p,float(x[0]),float(y[0]),float(z[0]))
    return error, [q1_p,q2_p,q3_p,q4_p]


def get_best_solution (end_point, model_list, model_number):
    # end_point: [x,y,z,pitch,yaw]
    # angles: [q1,q2,q3,q4]
    best_error=-1
    best_solution=[]
    if not (model_number>=0 and model_number<=2*Last_model+2):
        print("Warning in get_best_solution function: number of models to be used is wrong: " + str(model_number))
        return best_solution, best_error
    if not (len (model_list) > 0):
        print("Warning in get_best_solution function: model_list is empty")
        return best_solution, best_error
    # Fisrt model of "model_list" has the best initial solution:
    first_model=model_list[0]
    best_error, best_solution = get_error (end_point, first_model)
    best_model = 0
    # Check rest of the models for a better solution:
    my_model_range=range(1,model_number)
    for i in my_model_range:
        current_model=model_list[i]
        current_error, current_solution = get_error (end_point, current_model)
        if current_error < best_error:
            best_error = current_error
            best_solution = current_solution
            best_model = i
    return best_error, best_solution, best_model



def get_model_error_vector (test_samples_size, dataMat, model_number, model_list, directory, error_data_file, model_number_data_file):
    error_vector=np.zeros(test_samples_size,dtype=float)
    model_number_vector=np.zeros(test_samples_size,dtype=int)
    if os.path.isfile(directory+error_data_file) and os.path.isfile(directory+model_number_data_file):
        # Case 1) The training data was created before:
        error_vector=np.loadtxt(directory+error_data_file, dtype=float)
        model_number_vector=np.loadtxt(directory+model_number_data_file, dtype=int)
        return error_vector, model_number_vector
    # Case 2) The error matrix does not exist:
    if not os.path.isdir(directory):
        os.mkdir(directory)        
    # Error matrix creation
    data_input = dataMat[:,[4,5,6,7,8]]    # X,Y,Z,O_p,O_y
    #data_output = dataMat[:,[0,1,2,3]]    # Q1,Q2,Q3,Q4
    x = data_input[:,[0]] 
    y = data_input[:,[1]] 
    z = data_input[:,[2]] 
    pitch = data_input[:,[3]] 
    yaw = data_input[:,[4]] 
    # Create the error matrix:
    print("Begin models error vector construction: ")
    for row in range (0,test_samples_size):
        end_point=[x[row],y[row],z[row],pitch[row],yaw[row]]
        best_error, best_solution, best_model = get_best_solution (end_point, model_list, model_number)
        error_vector[row]=best_error
        model_number_vector[row]=best_model
        if (row%100==0):
            print("\t data test number (row): "+ str(row) + " processed")
    # Write the error vector:
    print("Begin model error and model number vectors disk writting: ")
    np.savetxt(directory+error_vector_model_prefix+ str(model_number+0)+".txt", error_vector,fmt='%f')
    np.savetxt(directory+number_vector_model_prefix+ str(model_number+0)+".txt", model_number_vector,fmt='%d')
    return error_vector, model_number_vector


def get_models_error_matrix (test_samples_size, dataMat, model_list, directory, error_data_file):
    error_matrix=np.zeros((test_samples_size,Last_model+1),dtype=float,order='F')
    column_range=range(0,Last_model+1)
    if os.path.isfile(directory+error_data_file):
        # Case 1) The training data was created before:
        with open(directory+error_data_file, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in reader:
                for column in column_range:
                    error_matrix[line_count][column]=float(row[column])
                line_count += 1
        return error_matrix
    # Case 2) The error matrix does not exist:
    if not os.path.isdir(directory):
        os.mkdir(directory)        
    # Error matrix creation
    data_input = dataMat[:,[4,5,6,7,8]]    # X,Y,Z,O_p,O_y
    #data_output = dataMat[:,[0,1,2,3]]    # Q1,Q2,Q3,Q4"error_vector_model_
    x = data_input[:,[0]] 
    y = data_input[:,[1]] 
    z = data_input[:,[2]] 
    pitch = data_input[:,[3]] 
    yaw = data_input[:,[4]] 
    # Create the error matrix:
    print("Begin models error matrix construction: ")
    for column in column_range:
        for row in range (0,test_samples_size):
            end_point=[x[row],y[row],z[row],pitch[row],yaw[row]]
            best_error, best_solution, best_model = get_best_solution (end_point, model_list, column)
            error_matrix[row][column]=best_error
            print("\t column/model: "+str(column) + " row: "+str(row))
        print ("get_models_error_matrix, column/model (row: "+str(row)+"): "+str(column))
    # Write the error matrix:
    print("Begin models error matrix disk writting: ")
    file = open (directory + error_data_file,"w")            
    for row in range (0,test_samples_size):
        for column in column_range:
            file.write(str(error_matrix[row][column]))
            file.write(",")
        file.write("\n")
    return error_matrix


def test_bootstrapping_approach (test_samples_size, dataMat, model_list, directory):
    model_number = len(model_list)
    if not os.path.isdir(directory):
        os.mkdir(directory)   
    error_data_file="error_vector_model_" + str(model_number+0) + ".txt"
    model_number_data_file=number_vector_model_prefix + str(model_number+0) + ".txt"
    error_vector, model_number_vector = get_model_error_vector (test_samples_size, dataMat, model_number, model_list, directory, error_data_file, model_number_data_file)
    # Plot results:
    mean=statistics.mean(error_vector)
    std_dev=statistics.pstdev(error_vector)
    quantiles=statistics.quantiles(error_vector, n = 4)
    print ('Test sample size: ' + str(test_samples_size))
    print ('mean: ' + str(mean))
    print ('std dev: ' + str(std_dev))
    print ('quantiles: ' + str(quantiles[0]) +' '+ str(quantiles[1]) + ' ' + str(quantiles[2]) )
    file = open (directory + "statistics.txt","w")   
    file.write ('Test sample size: ' + str(test_samples_size))
    file.write ("\n")
    file.write ('mean: ' + str(mean))
    file.write ("\n")
    file.write ('std dev: ' + str(std_dev))
    file.write ("\n")
    file.write ('quantiles: ' + str(quantiles[0]) +' '+ str(quantiles[1]) + ' ' + str(quantiles[2]) )
    file.write ("\n\n")
    #file.close

    angles_scatter_plot (dataMat,model_list,model_number_vector,directory,"108")
    angles_hexbin_plot (dataMat,model_list,model_number_vector,directory,"108")
    density_error_plot (test_samples_size, error_vector, mean, directory, str(model_number))
    histogram_and_density_error_plot (test_samples_size, error_vector, mean, directory, str(model_number))
    histogram_error_plot (test_samples_size, error_vector, mean, directory, str(model_number))
    threeD_error_plot (error_vector, dataMat, directory, str(model_number))
    threeD_error_plot_first_quadrant (error_vector, dataMat, directory, str(model_number))
    

if __name__ == '__main__':
    model_list=[]
    
    # Step 1: load MLP-ANN (models) of the sub-working area spaces directory:
    model_list1 = load_models (directory1, model_name_prefix_swas, Last_model+1)
    if len(model_list1) == 0:
      print('Sub-working area space model_list is empty!!')
      print ("Please, run before the scorbot_make_models_inverse_kinematics_sub_working_area_spaces.py to make the models")
      sys.exit(-1)
      
    # Step 2: load MLP-ANN (models) of the bootstrapping directory:
    model_list2 = load_models (directory2, model_name_prefix_bootstrapping, Last_model+1)
    if len(model_list2) == 0:
      print('Bootstrapping model_list is empty!!')
      print ("Please, run before the scorbot_make_models_inverse_kinematics.py to make the models")
      sys.exit(-1)
      
    # Step 3: load test file
    samples, dataMat = data_set_load (directory1, test_data_file_name)
    if len(dataMat) == 0:
      print('Test data file ' + test_data_file_name + ' is not found in directory ' + directory + '!!')
      sys.exit(-2)
    if samples != test_samples_size:
      print('Number of samples read: ' + samples + ' is not the same as ' + test_samples_size + '!!')      
      sys.exit(-3)
        
    # Set 4: get only vector error and vector_model (best model number for the error i-th)
    # This step is included in Step 5 in "test_bootstrapping_approach" function.
    #error_data_file=error_vector_model_prefix + str(Last_model+1) + ".txt"
    #model_number_data_file=number_vector_model_prefix + str(Last_model+1) + ".txt"
    #get_model_error_vector (test_samples_size, dataMat, 2*Last_model+2, model_list1 + model_list2, result_directory, error_data_file, model_number_data_file)

    # Step 5: get results
    test_bootstrapping_approach (test_samples_size, dataMat, model_list1 + model_list2, result_directory)
        
