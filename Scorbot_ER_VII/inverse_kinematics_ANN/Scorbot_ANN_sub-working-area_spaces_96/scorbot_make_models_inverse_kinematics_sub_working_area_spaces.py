import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import layers
#import statistics
from scorbot_common_functions_inverse_kinematics_sub_working_area_spaces import *

training_data_size= 4000
#samples = int (math.ceil(training_data_size/0.7))
number_epochs = 100



def build_and_train_general_model(model_name,dataMat):              # NN (MLP) Model
    if os.path.isfile(directory+model_name):
        print('Scorbot General Model '+ model_name + ' already exists. Loaded. No build process is performed ...')
        model = tf.keras.models.load_model(directory + model_name)
        return model

    ################### First step: BUILD MODEL (best values from previous tune processes are used)
    inputX = keras.Input(shape=(1,), name="Xcoor")
    inputY = keras.Input(shape=(1,), name="Ycoor")
    inputZ = keras.Input(shape=(1,), name="Zcoor")
    inputOp = keras.Input(shape=(1,), name="Opitch")
    inputOy = keras.Input(shape=(1,), name="Oyaw")
   
    # Base/q1 subnet:
    base_layer = layers.Dense(16, activation="sigmoid", name="base_layer") (inputOy)
    #base_layer = layers.Dense(50, activation="relu", name="base_layer2") (base_layer)
    base_q1 = layers.Dense(1, activation = "linear", name="q1")(base_layer)
    
    #Shoulder/q2 subnet:
    shoulder_input = layers.concatenate([inputX, inputZ, inputOp], name="shoulder_input")
    shoulder_layer = layers.Dense(272, activation="relu", name="shoulder_layer") (shoulder_input)
    #shoulder_layer = layers.Dense(50, activation="relu", name="shoulder_layer2") (shoulder_layer)
    shoulder_q2 = layers.Dense(1, activation = "linear", name="q2")(shoulder_layer)
   
    #Elbow/q3 subnet:
    elbow_input = layers.concatenate([inputX, inputZ, inputOp, shoulder_q2], name="elbow_input")    
    elbow_layer = layers.Dense(496, activation="relu", name="elbow_layer") (elbow_input)
    #elbow_layer = layers.Dense(50, activation="relu", name="elbow_layer2") (elbow_layer)
    elbow_q3 = layers.Dense(1, activation = "linear", name="q3")(elbow_layer)

    #Pitch or Wrist/q4 subnet:
    pitch_input = layers.concatenate([inputX, inputZ, inputOp, elbow_q3], name="pitch_input")
    pitch_layer = layers.Dense(496, activation="relu", name="pitch_layer") (pitch_input)
    #pitch_layer = layers.Dense(50, activation="relu", name="pitch_layer2") (pitch_layer)
    pitch_q4 = layers.Dense(1, activation = "linear", name="q4")(pitch_layer)

    model = keras.Model (inputs=[inputX, inputY, inputZ, inputOp, inputOy], outputs=[base_q1, shoulder_q2, elbow_q3, pitch_q4], name="ScorbotNN")
    
    #model.summary()
    #keras.utils.plot_model(model, directory+"scorbot_model_with_shape_info.png", show_shapes=True)
    
    optimizer = tf.keras.optimizers.Adam(0.01)
    model.compile(optimizer=optimizer,loss=keras.losses.mean_squared_error, metrics=['mse'])
    
    ################### Second step: TRAIN MODEL
    data_input = dataMat[:,[4,5,6,7,8]]    # X,Y,Z,O_p,O_y
    data_output = dataMat[:,[0,1,2,3]]    # Q1,Q2,Q3,Q4
    train_input = data_input[0:int(0.7*training_data_size),:]   # Separate data set in to Train, 
    train_output = data_output[0:int(0.7*training_data_size),:]
    test_input = data_input[int(0.7*training_data_size):training_data_size,:]
    test_output = data_output[int(0.7*training_data_size):training_data_size,:]

    x = train_input[:,[0]] 
    y = train_input[:,[1]] 
    z = train_input[:,[2]] 
    op = train_input[:,[3]] 
    oy = train_input[:,[4]] 
    q1_ = train_output[:,[0]] 
    q2_ = train_output[:,[1]] 
    q3_ = train_output[:,[2]] 
    q4_ = train_output[:,[3]] 

    x_t = test_input[:,[0]] 
    y_t = test_input[:,[1]] 
    z_t = test_input[:,[2]] 
    op_t = test_input[:,[3]] 
    oy_t = test_input[:,[4]] 
    q1_t = test_output[:,[0]] 
    q2_t = test_output[:,[1]] 
    q3_t = test_output[:,[2]] 
    q4_t = test_output[:,[3]] 

    name = "Model_Tracking"
    callbacks_list = [
            keras.callbacks.EarlyStopping (
                    monitor='val_loss',
                    patience=4,
                ),
                keras.callbacks.ReduceLROnPlateau (
                    factor=0.1,  
                    patience=3
                ),
                keras.callbacks.TensorBoard (
                    log_dir="logs/{}".format(name)      
                )
        ]

    model.fit ( {'Xcoor':x,'Ycoor':y,'Zcoor':z, 'Opitch':op, 'Oyaw':oy}, {'q1':q1_,'q2':q2_,'q3':q3_,'q4':q4_}, epochs=number_epochs, callbacks=callbacks_list, validation_data=([x_t,y_t,z_t,op_t,oy_t],[q1_t,q2_t,q3_t,q4_t]) )    #train the model
    model.save (directory+ model_name)
    
    return model


def build_and_train_pitch_model(model_name,dataMat):              # NN (MLP) Model to calculate "pitch angle"
    if os.path.isfile(directory+model_name):
        print('Scorbot Pitch Model '+model_name + ' already exists. Loaded. No build process is performed ...')
        model = tf.keras.models.load_model(directory + model_name)
        return model

    inputX = keras.Input(shape=(1,), name="Xcoor")
    inputZ = keras.Input(shape=(1,), name="Zcoor")
   
    pitch_input = layers.concatenate([inputX, inputZ], name="pitch_input")
    pitch_layer = layers.Dense(1008, activation="relu", name="pitch_layer1") (pitch_input)
    pitch_layer = layers.Dense(1008, activation="relu", name="pitch_layer2") (pitch_layer)
    pitch_angle = layers.Dense(1, activation = "linear", name="Opitch")(pitch_layer)

    model = keras.Model (inputs=[inputX, inputZ], outputs=[pitch_angle], name="ScorbotNN_pitch")
    
    #model.summary()
    #keras.utils.plot_model(model, directory+"scorbot_pitch_angle_model_with_shape_info.png", show_shapes=True)
    
    optimizer = tf.keras.optimizers.Adam(0.0037)
    model.compile(optimizer=optimizer,loss=keras.losses.mean_squared_error, metrics=['mse'])

    ################### Second step: TRAIN MODEL
    data_input = dataMat[:,[4,5,6,7,8]]    # X,Y,Z,O_p,O_y
    train_input = data_input[0:int(0.7*training_data_size),:]   # Separate data set in to Train, 
    test_input = data_input[int(0.7*training_data_size):training_data_size,:]

    x = train_input[:,[0]] 
    z = train_input[:,[2]] 
    op = train_input[:,[3]] 

    x_t = test_input[:,[0]] 
    z_t = test_input[:,[2]] 
    op_t = test_input[:,[3]] 

    name = "Model_Tracking"
    callbacks_list = [
            keras.callbacks.EarlyStopping (
                    monitor='val_loss',
                    patience=4,
                ),
                keras.callbacks.ReduceLROnPlateau (
                    factor=0.1,  
                    patience=3
                ),
                keras.callbacks.TensorBoard (
                    log_dir="logs/{}".format(name)      
                )
        ]

    model.fit ( {'Xcoor':x,'Zcoor':z}, {'Opitch':op}, validation_data=([x_t,z_t], [op_t]), epochs=number_epochs, callbacks=callbacks_list, verbose=0)
    model.save (directory+ model_name)
    
    return model


if __name__ == '__main__':

    if not os.path.isdir(directory):
      os.mkdir(directory)

    model_number = 0
    for t1 in range (1,t1_areas+1):         # t1 --> base
      for t2 in range (1,t2_areas+1):       # t2 --> shoulder
        for t3 in range (1,t3_areas+1):     # t3 --> elbow
          for t4 in range (1,t4_areas+1):   # t4 --> wrist
            # Open/load or create dataMat (data Set):
            training_data_name = training_data_prefix + str(model_number) + ".csv"
            print(training_data_name)
            samples_, dataMat = data_set_creation(training_data_size, training_data_name, t1, t2, t3, t4)
            # Open/load or build general MLP-ANN model:
            model_name = model_name_prefix + str(model_number) + ".h5"
            print(model_name)
            model = build_and_train_general_model(model_name,dataMat) 
            # MLP-ANN are numbered starting at 0
            model_number = model_number + 1 


        # Build pitch MLP:
        #pitch_model_name = pitch_model_name_prefix + str(model_number) + ".h5"
        #print(pitch_model_name)
        #model = build_and_train_pitch_model(pitch_model_name,dataMat)  
   
    
