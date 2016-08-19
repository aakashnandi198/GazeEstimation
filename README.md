# GazeEstimation
Visible spectrum,Non intrusive,Low cost,Gaze estimation

This program is a college project with the aim of creating a low cost gaze estimation.
The program is written in c++ using opencv 2.4.9 libraries.

The calibration folder contains "calibrator.cpp" which is used to collect generate data which when fed to polyfit function, 
gives the coefficient of the quadratic equation used to map the pupil coordinates with the coordinates on the screen.

The gaze folder contains "calibrator.cpp" is the actual gaze estimation code. It already has the coefficients hardcoded in it
and hence the code will run for screen size of 13" to 15".

Please find the video of the implementation runtime at :
https://www.youtube.com/watch?v=PmJ6Za4cD-s
