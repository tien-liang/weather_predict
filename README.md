# weather_predict

##Program description

Given a database of webcam images and a database of weather information, the machine will be
trained for the ability to predict weather with given image.

## Required Library
* I/O library used: sys, glob, shutil, os
* Open-cv 
* Numpy
* Pandas
* Sklearn
* Matplotlib


## Input files
* directory of csv files containing weather data
	* header should be at the 15th row
	* should include columns 'Year','Month','Day','Time','Weather'
* directory of jpg files 
	* file name should be in the format "katkam-yyyymmddhh0000.jpg" (yyyy = year, mm = month, dd = day, hh = hour)
* directory of jpg files for weather predicting
	* file name should include "yyyymmdd" so the software can recognized the date

##How To Run Program
* COMMAND: python3 weather\_prediction.py weather-directory image-directory test-image-directory
* Example command using provided files: python3 weather\_prediction.py yvr-weather katkam-scaled katkam-test
* Add test image into test-image-directory if want to test more image


##Output
* program will output an image window with weather description printed on the image
![](/image_0.jpg)