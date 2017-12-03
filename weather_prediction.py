import sys
import numpy as np
import pandas as pd
import glob
import cv2
import shutil
import os
import random
import re
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#=========================Function=========================#
#adjust number to 2 digits
def adjust(value):	
	return "%02i" % value
#extract hour from time
def extract_hour(time):
	return time[:2]
#get corresponding image from given filename
def get_image(filename):
	image = cv2.imread(filename,1)
	if image is not None:
		return image
	else:
		return np.NaN
#calculate image darkness
def average_darkness(image):
	return np.mean(image)
#calculate image color
def average_rgb(image):
	return np.mean(image,axis=0).mean(axis=0)
#calculate number of white pixels
def white(image):
		#reference for counting color pixels: https://stackoverflow.com/questions/32590932/count-number-of-black-pixels-in-an-image-in-python-with-opencv
	image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);
	ret,thresh = cv2.threshold(image,200,255,cv2.THRESH_BINARY)
	return cv2.countNonZero(thresh)
#calculate number of blue pixels
def blue(image):
	image = cv2.inRange(image, np.array([180,120,95]), np.array([255,210,180]))
	return cv2.countNonZero(image)

def extract_image(filename,des):
	shutil.copy(filename, des)
def rename_image(newname):
	filename = newname.split()
	os.rename(filename[0], newname)
def copy_dark(filename):
	if not os.path.exists('katkam-label/Dark_Image/'):
		os.mkdir('katkam-label/Dark_Image/')
	shutil.copy(filename, 'katkam-label/Dark_Image/')
	os.remove(filename)	
def copy_clear(filename):
	if not os.path.exists('katkam-label/clear_Image/'):
		os.mkdir('katkam-label/clear_Image/')
	shutil.copy(filename, 'katkam-label/clear_Image/')
def copy_clear(filename):
	if not os.path.exists('katkam-label/clear_Image/'):
		os.mkdir('katkam-label/clear_Image/')
	shutil.copy(filename, 'katkam-label/clear_Image/')
def copy_cloudy(filename):
	if not os.path.exists('katkam-label/cloudy_Image/'):
		os.mkdir('katkam-label/cloudy_Image/')
	shutil.copy(filename, 'katkam-label/cloudy_Image/')
def weather_cat(weather):
	if "Rain" in weather:
		weather = "Rain"
	elif "Clear" in weather:
		weather = "Clear"
	elif "Snow" in weather:
		weather = "Snow"
	elif "Cloudy" in weather:
		weather = "Cloudy"
	elif "Drizzle" in weather:
		weather = "Rain"
	else:
		weather = ""
	return weather
#============================================================#



#====================csv file to dataframe====================#
#read all weather csv files
csv_list = []
for file in glob.glob("yvr-weather/*.csv"):
	df = pd.read_csv(file,header=14)
	csv_list.append(df)
#concatenate all the dataframes into one dataframe
weather_data = pd.concat(csv_list)
#extract needed columns
weather_csv = weather_data[['Year','Month','Day','Time','Weather']]
#weather_csv['T'] = weather_data.iloc[:,6]
#weather_csv['H'] = weather_data.iloc[:,10]
#drop rows with no weather description
weather_csv = weather_csv.dropna()
#============================================================#



#=======================Get Image data=======================#
#assign image filename to available weather data
weather_csv['Month'] = weather_csv['Month'].apply(adjust)
weather_csv['Day'] = weather_csv['Day'].apply(adjust)
weather_csv['Time'] = weather_csv['Time'].apply(extract_hour)
weather_csv['imgfilename'] = 'katkam-scaled/katkam-' + weather_csv['Year'].map(str) + weather_csv['Month'].map(str) + weather_csv['Day'].map(str) + weather_csv['Time'].map(str) + '0000.jpg'
#read image of corresponding filename
weather_csv['image'] = weather_csv['imgfilename'].apply(get_image)
#drop rows with no image available
weather_csv = weather_csv.dropna()
#============================================================#



#===================Collect image features===================#
#image features
weather_csv['darkness'] = weather_csv['image'].apply(average_darkness)
weather_csv['rgb'] = weather_csv['image'].apply(average_rgb)
weather_csv[['R','G','B']] = pd.DataFrame(weather_csv.rgb.values.tolist(),index = weather_csv.index)
weather_csv = weather_csv.drop(['rgb'],axis=1)
weather_csv['white'] = weather_csv['image'].apply(white)
weather_csv['blue'] = weather_csv['image'].apply(blue)
weather_csv['white_blue_ratio'] = weather_csv['white']/weather_csv['blue']
weather_csv['white_blue_ratio'] = weather_csv['white_blue_ratio'].replace(np.inf,99999)
weather_csv['white_blue_ratio'] = weather_csv['white_blue_ratio'].replace(np.NaN,0)
weather_csv['rgb_difference'] = abs(weather_csv['R']-weather_csv['G'])+abs(weather_csv['B']-weather_csv['G'])
#Combine weather description into categories "Clear" "Mainly Clear" "Cloudy" "Mostly Cloudy" "Rain" "Fog" "Snow"
weather_csv['weather'] = weather_csv['Weather'].apply(weather_cat)
#Fix weather description of image with Clear sky 
weather_csv.loc[(weather_csv['blue']>10000) & (weather_csv['white_blue_ratio']<0.5) & (weather_csv['rgb_difference']>29),'weather'] = "Clear"
weather_csv['weather'] = weather_csv['weather'].replace('',np.NaN)
weather_csv = weather_csv.dropna()
#============================================================#



#==================Test images processing====================#
img_test = pd.DataFrame(columns=['Month','image'])
for file in glob.glob('katkam-test/*.jpg'): 
    img=cv2.imread(file,1)
    r = re.search('\d{10}',file)
    img_test.loc[len(img_test)]=[r.group(0)[4:6],img]
img_test.to_csv('test.csv')
img_test['darkness'] = img_test['image'].apply(average_darkness)
img_test['rgb'] = img_test['image'].apply(average_rgb)
img_test[['R','G','B']] = pd.DataFrame(img_test.rgb.values.tolist(),index = img_test.index)
img_test = img_test.drop(['rgb'],axis=1)
img_test['white'] = img_test['image'].apply(white)
img_test['blue'] = img_test['image'].apply(blue)
img_test['white_blue_ratio'] = img_test['white']/img_test['blue']
img_test['white_blue_ratio'] = img_test['white_blue_ratio'].replace(np.inf,99999)
img_test['white_blue_ratio'] = img_test['white_blue_ratio'].replace(np.NaN,0)
img_test['rgb_difference'] = abs(img_test['R']-img_test['G'])+abs(img_test['B']-img_test['G'])
#============================================================#



#===============Data cleaning and preprocessing================#
'''
#----------data cleaning (images)----------#
#comment out if no needed to run faster
#move the images with weather description available to another dir
des = 'katkam-scaled-n/'
if os.path.exists(des):	#if dir exist, delete
	shutil.rmtree(des)	
os.mkdir(des)	#make new dir
weather_csv['imgfilename'].apply(extract_image, des=des)	#copy needed images to new dir
shutil.rmtree('katkam-scaled/')	#remove original dir
os.rename(des,'katkam-scaled/')	#rename new dir
weather_csv[weather_csv['darkness']<65]['imgfilename'].apply(copy_dark)
'''
#weather_csv[(weather_csv['blue']>10000) & (weather_csv['white_blue_ratio']<0.5) & (weather_csv['rgb_difference']>29)]['imgfilename'].apply(copy_clear)
#weather_csv[weather_csv['weather']=='Clear']['imgfilename'].apply(copy_clear)
#weather_csv[weather_csv['weather']=='Cloudy']['imgfilename'].apply(copy_cloudy)
#------------------------------------------#



#-----data cleaning (weather description)-----#
'''
print(weather_csv.Weather.unique())
 'Mainly Clear' 'Clear' 
 'Mostly Cloudy' 'Cloudy' 
 'Rain' 'Moderate Rain' 'Heavy Rain'
 'Rain Showers' 'Moderate Rain Showers' 
 'Rain,Fog' 'Moderate Rain,Fog' 'Heavy Rain,Fog'
 'Rain,Drizzle' 'Moderate Rain,Drizzle'
 'Rain,Snow'
 'Rain Showers,Fog' 'Moderate Rain Showers,Fog' 'Thunderstorms' 
 'Rain,Drizzle,Fog'
 'Rain,Snow,Fog'
 'Fog' 'Freezing Fog'
 'Drizzle' 'Drizzle,Fog'
 'Snow' 'Snow Showers' 'Moderate Snow'
 'Snow,Fog'
 'Rain Showers,Snow Showers,Fog' 'Rain Showers,Snow Pellets'
 'Rain Showers,Snow Showers' 'Heavy Rain Showers,Moderate Snow Pellets,Fog'
 'Heavy Rain,Moderate Hail,Fog'
'''

#print(weather_csv.groupby(['Weather'])['Weather'].count())
'''Clear                                           256
Cloudy                                          606
Drizzle                                          18
Drizzle,Fog                                       9
Fog                                              18
Heavy Rain                                        1
Heavy Rain,Fog                                    2
Mainly Clear                                    532
Moderate Rain                                    20
Moderate Rain Showers                             2
Moderate Rain Showers,Fog                         2
Moderate Rain,Drizzle                             2
Moderate Rain,Fog                                11
Moderate Snow                                     4
Mostly Cloudy                                   572
Rain                                            538
Rain Showers                                    156
Rain Showers,Fog                                  7
Rain Showers,Snow Showers                         1
Rain Showers,Snow Showers,Fog                     1
Rain,Drizzle                                      4
Rain,Drizzle,Fog                                  5
Rain,Fog                                         91
Rain,Snow                                         7
Rain,Snow,Fog                                     1
Snow                                             61
Snow Showers                                     10
Snow,Fog                                          5
'''
#put weather description on filename to do some check on fitting b/w descriptions and images
#weather_csv['new_filename'] = weather_csv['imgfilename']+ ' ' + weather_csv['Weather']
#weather_csv['new_filename'].apply(rename_image)

#print(weather_csv[weather_csv['Weather']=='Thunderstorms']['imgfilename']) #only 2 files so removed
#print(weather_csv[weather_csv['Weather']=='Freezing Fog']['imgfilename'])	#only 2 files so removed
#print(weather_csv[weather_csv['Weather']=='Heavy Rain,Moderate Hail,Fog']['imgfilename'])	#only 1 files so removed
#print(weather_csv[weather_csv['Weather']=='Heavy Rain Showers,Moderate Snow Pellets,Fog']['imgfilename'])	#only 1 files so removed
#print(weather_csv[weather_csv['Weather']=='Rain Showers,Snow Pellets']['imgfilename'])	#only 1 files so removed
#write to csv
#weather_csv.to_csv('weather.csv')
#============================================================#



#=======================Model training=======================#
X = weather_csv[['Month','darkness','R','G','B','white','blue','white_blue_ratio','rgb_difference']].values
y = weather_csv['weather'].values
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
#model = KNeighborsClassifier(n_neighbors=20)
X_test = img_test[['Month','darkness','R','G','B','white','blue','white_blue_ratio','rgb_difference']].values
model = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear',C=6)
)
model.fit(X, y)
#y_predicted = model.predict(X_test)
#print(accuracy_score(y_test, y_predicted))



for x in range(0,len(X_test)):
	#r = random.randint(0,2700)
	#data = weather_csv.iloc[r]
	predicted = model.predict(X_test[x].reshape(1,-1))
	img = img_test.iloc[x]['image']
	#if data['weather'] != predicted[0]:
	#	cv2.putText(img,predicted[0],(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
	#	print("Actual weather: "+data['weather'])
	#else:
	cv2.putText(img,predicted[0],(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
	cv2.imshow('image',img)
	cv2.imwrite('image_'+str(x)+'.jpg',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
#============================================================#