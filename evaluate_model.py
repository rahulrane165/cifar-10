# Python scipt to evaluate CIFAR-10 model

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os
import keras
from keras.models import load_model

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

def normalize_data(train, test):
	train = train.astype('float32')
	test = test.astype('float32')
	train /= 255.0
	test /= 255.0
	return train, test

if __name__ == "__main__":
	"""Main function to evaluate model"""

	num_classes = 10
	#Path to directory
	cifar_10_dir = './cifar-10-batches-py'

	#Load data function
	(x_train, y_train), (x_test, y_test) = load_data(cifar_10_dir)

	# Convert class vectors to binary class matrices.
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	#Normalize the image data
	x_test, y_test = normalize_data(x_test, y_test)

	#Load model
	model = load_model('saved_models/cifar-10_model.h5')

	#Evaluate model on test dataset
	_, acc = model.evaluate(x_test, y_test, verbose=0)
	print('Model evaluation result : %.3f' % (acc * 100.0))

