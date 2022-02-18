# Python scipt to build/train CIFAR-10 model

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
import pickle
import sys
import os

def load_batch(path, label_key='labels'):
	""" CIFAR-10 data.
	# Arguments
		path: path the file to parse.
		label_key: key for label data in the retrieve dictionary.
	# Returns
		A tuple `(data, labels)`.
	"""
	with open(path, 'rb') as f:
		if sys.version_info < (3,):
			d = pickle.load(f)
		else:
			d = pickle.load(f, encoding='bytes')
			# decode utf8
			d_decoded = {}
			for k, v in d.items():
				d_decoded[k.decode('utf8')] = v
			d = d_decoded
	data = d['data']
	labels = d[label_key]

	data = data.reshape(data.shape[0], 3, 32, 32)
	return data, labels

def load_data(path, negatives=False):
	"""Loads CIFAR-10 dataset.
	# Returns
		Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`
	"""

	num_train_samples = 50000

	x_train_local = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
	y_train_local = np.empty((num_train_samples,), dtype='uint8')

	for i in range(1, 6):
		fpath = os.path.join(path, 'data_batch_' + str(i))
		(x_train_local[(i - 1) * 10000: i * 10000, :, :, :],
		y_train_local[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

	fpath = os.path.join(path, 'test_batch')
	x_test_local, y_test_local = load_batch(fpath)

	y_train_local = np.reshape(y_train_local, (len(y_train_local), 1))
	y_test_local = np.reshape(y_test_local, (len(y_test_local), 1))

	if negatives:
		x_train_local = x_train_local.transpose(0, 2, 3, 1).astype(np.float32)
		x_test_local = x_test_local.transpose(0, 2, 3, 1).astype(np.float32)
	else:
		x_train_local = np.rollaxis(x_train_local, 1, 4)
		x_test_local = np.rollaxis(x_test_local, 1, 4)

	return (x_train_local, y_train_local), (x_test_local, y_test_local)

if __name__ == "__main__":
	"""Main function to load data"""

	#Setting up Hyper-parameters

	batch_size = 32
	num_classes = 10
	epochs = 100
	data_augmentation = True
	num_predictions = 20
	save_dir = os.path.join(os.getcwd(), 'saved_models')
	model_name = 'cifar-10_model.h5'

	#Path to directory
	cifar_10_dir = './cifar-10-batches-py'

	#Load data function
	(x_train, y_train), (x_test, y_test) = load_data(cifar_10_dir)

	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	#Normalize the image data
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	# Convert class vectors to binary class matrices.
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	# define cnn model
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.3))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)

	# Train the model using SGD
	model.compile(loss='categorical_crossentropy',
				optimizer=opt,
				metrics=['accuracy'])

	if not data_augmentation:
		print('Not using data augmentation.')
		model.fit(x_train, y_train,
				batch_size=batch_size,
				epochs=epochs,
				validation_data=(x_test, y_test),
				shuffle=True)
	else:
		print('Using real-time data augmentation.')
		# This will do preprocessing and realtime data augmentation:
		datagen = ImageDataGenerator(
			featurewise_center=False,  # set input mean to 0 over the dataset
			samplewise_center=False,  # set each sample mean to 0
			featurewise_std_normalization=False,  # divide inputs by std of the dataset
			samplewise_std_normalization=False,  # divide each input by its std
			zca_whitening=False,  # apply ZCA whitening
			zca_epsilon=1e-06,  # epsilon for ZCA whitening
			rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
			# randomly shift images horizontally (fraction of total width)
			width_shift_range=0.1,
			# randomly shift images vertically (fraction of total height)
			height_shift_range=0.1,
			shear_range=0.,  # set range for random shear
			zoom_range=0.,  # set range for random zoom
			channel_shift_range=0.,  # set range for random channel shifts
			# set mode for filling points outside the input boundaries
			fill_mode='nearest',
			cval=0.,  # value used for fill_mode = "constant"
			horizontal_flip=True,  # randomly flip images
			vertical_flip=False,  # randomly flip images
			# set rescaling factor (applied before any other transformation)
			rescale=None,
			# set function that will be applied on each input
			preprocessing_function=None,
			# image data format, either "channels_first" or "channels_last"
			data_format=None,
			# fraction of images reserved for validation (strictly between 0 and 1)
			validation_split=0.0)

	# Compute quantities required for feature-wise normalization
	# (std, mean, and principal components if ZCA whitening is applied).
	datagen.fit(x_train)

	# Fit the model on the batches generated by datagen.flow().
	model.fit_generator(datagen.flow(x_train, y_train,
										batch_size=batch_size),
						steps_per_epoch=len(x_train)/batch_size,
						epochs=epochs,
						validation_data=(x_test, y_test),
						workers=4)

	# Save model and weights
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	model_path = os.path.join(save_dir, model_name)
	model.save(model_path)
	print('Saved trained model at %s ' % model_path)

