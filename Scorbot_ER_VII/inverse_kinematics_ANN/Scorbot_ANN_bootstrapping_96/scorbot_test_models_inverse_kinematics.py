from scorbot_common_functions_inverse_kinematics import *
from scorbot_plot_functions_inverse_kinematics import *
import tensorflow as tf
#from tensorflow import keras

test_samples_size_error_matrix=96
test_data_file_name_error_matrix="test_data_set_96.csv"
error_data_file_error_matrix="models_error_matrix_96.csv"

test_samples_size=5000
test_data_file_name="test_data_set.csv"
error_data_file="models_error_matrix.csv"

directory_model_0="model_0_results/"
directory_model_96="model_96_results/"


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
    if not (model_number>=First_model and model_number<=Last_model+1):
        print("Warning in get_best_solution function: number of models to be used is wrong: " + str(model_number))
        return best_solution, best_error
    if not (len (model_list) > 0):
        print("Warning in get_best_solution function: model_list is empty")
        return best_solution, best_error
    # Fisrt model of "model_list" has the best initial solution:
    first_model=model_list[0]
    best_error, best_solution = get_error (end_point, first_model)
    # Check rest of the models for a better solution:
    my_model_range=range(1,model_number)
    for i in my_model_range:
        current_model=model_list[i]
        current_error, current_solution = get_error (end_point, current_model)
        if current_error < best_error:
            best_error = current_error
            best_solution = current_solution
    return best_error, best_solution



def get_model_error_vector (test_samples_size, dataMat, model_number, model_list, directory, error_data_file):
    error_vector=np.zeros(test_samples_size,dtype=float)
    if os.path.isfile(directory+error_data_file):
        # Case 1) The training data was created before:
        error_vector=np.loadtxt(directory+error_data_file, dtype=float)
        return error_vector
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
        best_error, best_solution = get_best_solution (end_point, model_list, model_number)
        error_vector[row]=best_error
        if (row%100==0):
            print("\t data test number (row): "+ str(row) + " processed")
    # Write the error vector:
    print("Begin model error vector disk writting: ")
    np.savetxt(directory+"error_vector_model_"+ str(model_number)+".txt", error_vector,fmt='%f')
    return error_vector


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
            best_error, best_solution = get_best_solution (end_point, model_list, column)
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


def test_bootstrapping_approach_with_N_models (test_samples_size, dataMat, model_number, model_list, directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)   
    error_data_file="error_vector_model_" + str(model_number) + ".txt"
    error_vector=get_model_error_vector (test_samples_size, dataMat, model_number, model_list, directory, error_data_file)
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

    if model_number==0:
        angles_scatter_plot (dataMat,model_list[0],directory,"0")
        angles_hexbin_plot (dataMat,model_list[0],directory,"0")
    density_error_plot (test_samples_size, error_vector, mean, directory, str(model_number))
    histogram_and_density_error_plot (test_samples_size, error_vector, mean, directory, str(model_number))
    histogram_error_plot (test_samples_size, error_vector, mean, directory, str(model_number))
    threeD_error_plot (error_vector, dataMat, directory, str(model_number))
    threeD_error_plot_first_quadrant (error_vector, dataMat, directory, str(model_number))
    

if __name__ == '__main__':
    model_list=[]
    
    # Step 1: get data set (dataMat) and load MLP (models):
    # Version 1: for the matrix_error_calculation to compare models (step 2)
    #samples_, dataMat = data_set_creation(test_samples_size_error_matrix, directory, test_data_file_name_error_matrix)
    # Version 2: to test a single model
    samples_, dataMat = data_set_creation(test_samples_size, directory, test_data_file_name)
    model_list=load_models (directory,model_name_prefix,Last_model+1)
    
    # Step 2 (optional): calculate and compare the errors of the whole models.
    # Bootstrapping with 1 model (single or classical MLP), 2 models, 3 models, etc.
    # Ths results help to decide what is the ideal number of models to use when
    # comparing with the classical MLP approach (1 single model).
    # Please take into account that the process is slow because the same data set (datamat)
    # must be tested first with 1 model, then with 2 models, then with 3 models, etc.
    #error_matrix=get_models_error_matrix(test_samples_size_error_matrix, dataMat, model_list, directory, error_data_file_error_matrix)
    #bar_error_matrix_plot (error_matrix, Last_model, directory)
    
    # Step 3: get results from the classical MLP that uses one model (one neural network).
    #test_bootstrapping_approach_with_N_models (test_samples_size, dataMat, 0, model_list, directory_model_0)
    
    # Step 4: get results from the bootstrapping approach that takes N models 
    # (neural networks) instead one 50 in this case). The objective is to compare 
	# with results of Step 3.
    test_bootstrapping_approach_with_N_models (test_samples_size, dataMat, 96, model_list, directory_model_96)
    
