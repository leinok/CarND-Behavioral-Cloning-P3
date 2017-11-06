# This is model.py which is a script used to create and train the model

import numpy as np
import os
import tensorflow as tf
import pdb
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Lambda, MaxPooling2D, Flatten, Dropout, Activation 
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

import matplotlib.image as mpimg

def loadData():
	csv_file = "~/Downloads/data/driving_log.csv"
	df = pd.read_csv(csv_file)
	X = df[['center', 'left', 'right']].values
	y = df['steering'].values

	X_train, X_test, y_train, y_test = \
				train_test_split(X, y, test_size = 0.2, random_state = 42)
	return X_train, X_test, y_train, y_test

# Load image
def loadImage(data_dir, image_file):
	image = cv2.imread(os.path.join(data_dir, image_file.strip()))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	return image

# Brightness
def augment_brightness(image):
    	HSV_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    	random_bright = 0.3 + np.random.uniform()
    	HSV_image[:,:,2] = HSV_image[:,:,2]*random_bright
    	image = cv2.cvtColor(HSV_image, cv2.COLOR_HSV2RGB)
	return image

# Remove the pixel information which does not contribute to the regression. (upper and lower)
def cropImage(image, lower = 0.35, upper = 0.85):
	lower_pos = int(lower * image.shape[0])
	upper_pos = int(upper * image.shape[0])
	return image[lower_pos:upper_pos, :, :]

# Preprocess: crop and resize
def preprocess(image):
	image = cropImage(image)
	image = cv2.resize(image, (200, 66), cv2.INTER_AREA)
	return image

# Random flip images
def flipImage(image, steering_angle):
	prob = 0.5
	if np.random.random() > prob:
		return cv2.flip(image, 1), steering_angle * (-1)
	
	else:
		return image, steering_angle

#------------------------------------------------------------------------------
# Using Python generator to genrate data
# Create training arrays to contain batch of images and steering angles
def generator(data_dir, image_paths, steering_angles, batch_size, augment_flag = False):
	images = np.empty([batch_size, 66, 200, 3])
	steers = np.empty(batch_size)
	while True:
		i = 0
		for index in np.random.permutation(image_paths.shape[0]):
			center, left, right = image_paths[index]
			steering_angle = steering_angles[index]
			# Load image
			random_value = np.random.random()
			if random_value > 0.67:
				image = loadImage(data_dir, left)
				steering_angle = steering_angle + 0.15
			elif random_value > 0.33:
				image = loadImage(data_dir, right)
				steering_angle = steering_angle - 0.15
			else:
				image = loadImage(data_dir, center) 
			# add the image and steering angle to the batch
			if augment_flag:
				image, steering_angle = flipImage(image, steering_angle)
				image = augment_brightness(image)
			image = preprocess(image)
			images[i] = image
			steers[i] = steering_angle
			i = i + 1
			if i == batch_size:
				break
		yield images, steers

def train_model(X_train, X_test, y_train, y_test):
	# Set the model architecture exactly with NVIDIA model.
	# Training Parameters, set the same parameters as the paper
	model = Sequential()
	# add a x -> x/127.5 layer, normalized input
	model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
	# starts with five convolutional and maxpooling layers with filter size 5 X 5
	model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	# 2 convolutional layers with filter size 3 X 3
	model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Flatten())

	model.add(Dense(1164))
	model.add(Activation('relu'))
	
	# I added dropout layer here
	model.add(Dropout(0.5))

	model.add(Dense(100))
	model.add(Activation('relu'))

	model.add(Dense(50))
	model.add(Activation('relu'))

	model.add(Dense(10))
	model.add(Activation('relu'))

	model.add(Dense(1))

	print(model.summary())

	# --------------------------------End of model architecture
	model.compile(loss='mse', optimizer=Adam(lr=1e-4))

	# Fit the model on data generated batch-by-batch by a Python generator
	# The generator is run in parallel to the model. Real-time data augmentation on images on CPU in parallel
	data_dir = '/home/saiclei/Downloads/data/'


	# Define two generators
	train_generator = generator(data_dir, X_train, y_train, 64, True)
	validation_generator = generator(data_dir, X_test, y_test, 64, False)
	
	model.fit_generator(train_generator,
		  		steps_per_epoch = 300,
				epochs = 6,
				max_queue_size = 1,
				validation_data = validation_generator,
				nb_val_samples = len(X_test),
				verbose = 1)
	model.save('model-final.h5')
	
if __name__ == "__main__":
	from tensorflow.python.client import device_lib
	print(device_lib.list_local_devices())
	data = loadData()
	train_model(*data)
