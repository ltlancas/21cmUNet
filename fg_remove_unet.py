# code for a UNet architecture for learning how to remove 
# the foreground from a 21ccm signal

from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def build_model():
	inputs = keras.layers.Input(shape=(64,64,30),name="image_input")
	conv1a = keras.layers.Conv2D(64,7,activation=tf.nn.relu)(inputs)
	conv1b = keras.layers.Conv2D(64,7,activation=tf.nn.relu)(conv1a)
	mp1 = keras.layers.MaxPool2D(2)(conv1b)
	conv2a = keras.layers.Conv2D(128,3,activation=tf.nn.relu)(mp1)
	conv2b = keras.layers.Conv2D(128,3,activation=tf.nn.relu)(conv2a)
	mp2 = keras.layers.MaxPool2D(2)(conv2b)
	conv3a = keras.layers.Conv2D(256,3,activation=tf.nn.relu)(mp2)
	conv3b = keras.layers.Conv2D(256,2,activation=tf.nn.relu)(conv3a)
	mp3 = keras.layers.MaxPool2D(2)(conv3b)
	# symmetric upsampling path
	upsamp2 = keras.layers.UpSampling2D(5)(mp3)
	upconv2 = keras.layers.Conv2DTranspose(128,3)(upsamp2)
	concat2 = keras.layers.concatenate([conv2b,upconv2])
	conv4a =  keras.layers.Conv2D(128,3,activation=tf.nn.relu)(concat2)
	conv4b =  keras.layers.Conv2D(128,3,activation=tf.nn.relu)(conv4a)
	upsamp1 = keras.layers.UpSampling2D(3)(conv4b)
	conv5 =  keras.layers.Conv2D(64,3,activation=tf.nn.relu)(upsamp1)
	concat1 = keras.layers.concatenate([conv1b,conv5])
	upconv1a = keras.layers.Conv2DTranspose(64,7)(concat1)
	upconv1b = keras.layers.Conv2DTranspose(64,7)(upconv1a)

	output = keras.layers.Conv2DTranspose(30,7)(upconv1a)

	model = keras.models.Model(inputs=inputs,outputs=output)
	return model

if __name__ == '__main__':
	# build model
	model = build_model()
	# ccompile model with specified loss and optimizer
	model.compile(optimizer="adam", loss="mse",metrics=["accuracy"])
	# give a summary of the model
	#print(model.summary())
	
	# load the training data
	data = np.load("/Volumes/My Passport for Mac/lachlanl/21cm_project/sims/observed_nsim100_fg1.npy")
	signal = np.load("/Volumes/My Passport for Mac/lachlanl/21cm_project/sims/cosmo_nsim100_fg1.npy")

	# split the data in to training/validation/testing sets
	x_train = data[:18000]
	y_train = signal[:18000]
	x_val = data[19000:-200]
	y_val = signal[19000:-200]
	x_test = data[-200:]
	y_test = signal[-200:]

	# train the model
	N_EPOCHS = 20
	N_BATCH = 64
	history = model.fit(x_train,y_train,batch_size=N_BATCH,epochs=N_EPOCHS,validation_data=(x_val, y_val))

	# save the results of the training
	model.save_weights("./model")