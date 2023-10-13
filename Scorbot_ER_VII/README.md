This code is related to paper:

Daniel Cagigas-Muñiz, "Artificial Neural Networks for inverse kinematics problem in articulated robots", Engineering Applications of Artificial Intelligence, Volume 126, Part D, 2023, 107175, ISSN 0952-1976, https://doi.org/10.1016/j.engappai.2023.107175.

This is software to simulate and calculate the kinematics (direct and indirect) of the Scorbot ER VII:

- <ins>scorbot_gazebo_tf</ins>: this is a ROS package to simulate the Scorbot ER VII robot. Both gazebo and rviz simulation frameworks are launched. There are two Python scripts to test the direct and inverse kinematics. Please, read the README.pdf for installing and launching the “scorbot_gazebo_tf” package in a Ubuntu computer.

- <ins>inverse_kinematics_ANN</ins>: Python scripts for calculating the Scorbot ER VII Inverse Kinematics using Artificial Neural Networks (ANNs). There are also plots, .txt results/output files, trained ANN in .h5 formats, .csv datasets, etc. The Python scripts can generate all of these files again. Please, read "ReadMe" files inside for more details.
