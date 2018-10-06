import gender.training_test_set as tts
import data_set as ds
from settings import MODEL_DIR_PATH
from callbacks.test_metrics import TestMetrics
from callbacks.csv_logger import MyCSVLogger

import numpy as np
import cv2
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras import backend as K

K.set_image_dim_ordering('th')

GENDER_MODEL_DIR_PATH = os.path.join(MODEL_DIR_PATH, 'gender')
GENDER_MODEL_PATH = os.path.join(GENDER_MODEL_DIR_PATH, 'model.h5')
GENDER_MODEL_HISTORY_DIR_PATH = os.path.join(GENDER_MODEL_DIR_PATH, 'history')

def create_model(input_shape, output_dim):
	"""
	Create a new model.

	:input_shape: The shape of the inputs.
	:output_dim: The dimension of the outputs.
	:return: A new model.
	"""

	# Create the model.
	model = Sequential()

	# Block 1.
	model.add(Conv2D(32, (3, 3), padding='same', use_bias=False, input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.4))

	# Block 2.
	model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.4))

	# Block 3.
	model.add(Conv2D(128, (5, 5), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.4))

	# Block 4.
	model.add(Conv2D(256, (7, 7), padding='same', use_bias=False))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Flatten())

	# Fully connected layers.
	model.add(Dense(128, use_bias=False))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	model.add(Dense(256, use_bias=False))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.3))

	model.add(Dense(512, use_bias=False))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.4))

	model.add(Dense(64, use_bias=False))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.1))
	
	model.add(Dense(output_dim, use_bias=False))
	model.add(BatchNormalization())
	model.add(Activation('softmax'))

	return model

def compile_model(model):
	"""
	Compile the model.

	:model: The model.
	"""

	optimizer = optimizers.Adam(lr=0.01, decay=1e-6)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

def preprocess_inputs(inputs):
	"""
	Preprocess the inputs for the neural network. This functions does three things: resize, reshape and normalize the
	inputs.
	1) Resize the inputs (images) (to 'data_set.IMAGE_WIDTH' X 'data_set.IMAGE_HEIGHT').
	2) Reshape the inputs into an array with four dimensions:
	- first dimension: instances;
	- second dimension: color dimension (data_set.IMAGE_COLOR_DIMENSION);
	- third dimension: image width (data_set.IMAGE_WIDTH);
	- fourth dimension: image height (data_set.IMAGE_HEIGHT).
	3) Normalize the inputs (from 0 - 255 to 0 - 1).

	:inputs: The inputs. It is a list of Numpy arrays. Each element in the list is an instance. Each instance (Numpy
		array) have three dimensions. The first one is the image height. The second one is the image width. And the last
		one is the color dimension (data_set.IMAGE_COLOR_DIMENSION).
	:return: The preprocessed inputs.
	"""

	# Resize the images.
	inputs = np.array([
		cv2.resize(ipt, (ds.IMAGE_WIDTH, ds.IMAGE_HEIGHT), interpolation=cv2.INTER_AREA) for ipt in inputs
	])

	# Reshape to be [samples] [pixels] [width] [height].
	# First dimension: instances.
	# Second dimension: color dimension (data_set.IMAGE_COLOR_DIMENSION).
	# Third dimension: image width (data_set.IMAGE_WIDTH).
	# Fourth dimension: image height (data_set.IMAGE_HEIGHT).
	inputs = inputs.reshape(inputs.shape[0], ds.IMAGE_COLOR_DIMENSION, ds.IMAGE_WIDTH,
		ds.IMAGE_HEIGHT).astype('float32')

	# Normalize inputs from 0 - 255 to 0 - 1.
	inputs = inputs / 255

	return inputs

def preprocess_outputs(outputs):
	"""
	Preprocess the outputs for the neural network. This function does only one thing: transform outputs to use the one
	hot encoding.

	:outputs: The outputs.
	:return: The preprocessed outputs.
	"""

	# Transform outputs to use the one hot encoding.
	outputs = np_utils.to_categorical(outputs)

	return outputs

if __name__ == '__main__':
	from settings import PLOT_DIR_PATH
	import data_set as ds

	import matplotlib.pyplot as plt
	
	# Fix random seed for reproducibility.
	seed = 42
	np.random.seed(seed)

	# Load the dataset.
	# First dimension: instances.
	# Second dimension: image height (data_set.IMAGE_HEIGHT).
	# Third dimension: image width (data_set.IMAGE_WIDTH).
	# Fourth dimension: color dimension (data_set.IMAGE_COLOR_DIMENSION).
	(x_train, y_train), _ = tts.load(test_set_proportion=0.0)
	_, x_test, y_test = tts.load_test_set(os.path.join(ds.TEST_SET_PATH, 'gender'))
	print("Data set loaded")

	# Reshape to be [samples] [pixels] [width] [height].
	# First dimension: instances.
	# Second dimension: color dimension (data_set.IMAGE_COLOR_DIMENSION).
	# Third dimension: image width (data_set.IMAGE_WIDTH).
	# Fourth dimension: image height (data_set.IMAGE_HEIGHT).
	# And normalize inputs from 0 - 255 to 0 - 1.
	x_train = preprocess_inputs(x_train)
	x_test = preprocess_inputs(x_test)

	# Transform outputs to use the one hot encoding.
	y_train = preprocess_outputs(y_train)
	y_test = preprocess_outputs(y_test)
	num_classes = y_test.shape[1]

	# Create and compile the model.
	model = create_model((ds.IMAGE_COLOR_DIMENSION, ds.IMAGE_WIDTH, ds.IMAGE_HEIGHT), num_classes)
	compile_model(model)
	#from keras.models import load_model
	#model = load_model(GENDER_MODEL_PATH)

	# Train the model.
	test_metrics = TestMetrics(x_test, y_test)
	checkpointer = ModelCheckpoint(filepath=os.path.join(GENDER_MODEL_HISTORY_DIR_PATH, 'model_ep{epoch:03d}.h5'),
		verbose=1)
	csv_logger = MyCSVLogger(os.path.join(GENDER_MODEL_HISTORY_DIR_PATH, 'metric_results.csv'), separator=';')
	callback_list = [test_metrics, checkpointer, csv_logger]
	history = model.fit(x_train, y_train, validation_split=0.05, epochs=100, batch_size=64, callbacks=callback_list)
	#history = model.fit(x_train, y_train, validation_split=0.05, epochs=100, batch_size=64, initial_epoch=8, callbacks=callback_list)

	# Save the model.
	model.save(GENDER_MODEL_PATH)

	# Evaluate the model.
	scores = model.evaluate(x_test, y_test, verbose=0)
	for name, score in zip(model.metrics_names, scores):
		print("{}: {:.3%}".format(name, score))
	print("Baseline Error: {:.3}%".format((100 - scores[1] * 100)))

	# Plots.
	# Compute the epoch numbers (abscissa), the training and test loss and the training and test accuracy (ordinate).
	epoch_numbers = history.epoch
	train_loss_results = history.history['loss']
	train_acc_results = history.history['acc']
	test_loss_results = history.history['test_loss']
	test_acc_results = history.history['test_acc']

	# Evolution of the training and test loss.
	fig = plt.figure(figsize=(8, 8))
	plt.plot(epoch_numbers, train_loss_results, color='red')
	plt.plot(epoch_numbers, test_loss_results, color='blue')
	plt.title('Evolution of the training and test loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend(['Training loss', 'Test loss'], loc='upper right')
	plt.tight_layout()
	plt.savefig(os.path.join(PLOT_DIR_PATH, 'gender', 'training_test_loss_evolution.png'), format='png', dpi=300)
	plt.show()

	# Evolution of the training and test accuracy.
	fig = plt.figure(figsize=(8, 8))
	plt.plot(epoch_numbers, train_acc_results, color='red')
	plt.plot(epoch_numbers, test_acc_results, color='blue')
	plt.title('Evolution of the training and test accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend(['Training accuracy', 'Test accuracy'], loc='lower right')
	plt.tight_layout()
	plt.savefig(os.path.join(PLOT_DIR_PATH, 'gender', 'training_test_acc_evolution.png'), format='png', dpi=300)
	plt.show()
