This directory/folder contains the Python scripts that predict the inverse kinematic of the Scorbot ER VII robot. The Spyder 3 programming framework was used to program this code. Please, use better this programming framework if you want to work with this code. Python 3.x version is used. Four methods are proposed:

1) <ins>Scorbot_ANN_bootstrapping_and Classical_approach</ins>: here the classical approach using one MLP-ANN (Multimultilayer Perceptron Artificial Neural Network). The tuning process is also included using Keras tuner. The tuning proccess here is the final result of other tests not included here: two layers, scaled inputs, etc. Only one hidden layer and not scaled input is considered. Also the bootstrap sampling (bagging) is considered. This method uses an ensemble of MLP-ANNs to compute the Scorbot ER VII inverse kinematics. The direct kinematics equations are included in file "scorbot_common_functions_inverse_kinematics_hybrid.py". Please, read "SCRIPTS_DESCRIPTION.pdf" that describes the Python scripts.

2) <ins>Scorbot_ANN_bootstrapping_96</ins>: the same bootstrap sampling technique as above but particularized to the use of 96 models.

3) <ins>Scorbot_ANN_sub-working-area_spaces_96</ins>: it divides the workspace into 96 areas and assigns a neural network to each of them.

4) <ins>Scorbot_ANN_hybrid_bootstrapping-swas_54+54</ins>: a combination of 2) and 3): 54 MLP-ANNs are used for each method (108 MLP-ANNs in total).

The MLP-ANNs (models) and data training/test sets are also included. They can be generated again if necessary. Using this data, plots and results can be obtained in a short execution period of time. However, some processes like the error vector generation (minimum error produced of the ensemble of ANNs for every data test point) can take some time if this was not previous calculated. This data is again included to run and obtain fast the plots/results.

The four methods have very similar Python scripts. The general structure and contents of these Python scripts are described in file: SCRIPTS_DESCRIPTION.pdf. Please, read this information before start working.
