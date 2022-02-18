
A Keras-TensorFlow Convolutional Neural Network used for training and testing on the CIFAR-10 dataset.

CIFAR10 dataset includes:

1. 50,000 32x32 color training images.
2. 10,000 32x32 color testing images.
3. 10 different classes.
4. 6,000 images per class.

Prerequisites:
Hardware Requirements:
	Development PC (OS:Ubuntu 18.04.6 LT)

Installations:
	Please follow below commands.

	$ pip3 install --upgrade pip3 setuptools
	$ git clone https://github.com/rahulrane165/cifar-10
	$ cd cifar-10/
	$ pip3 install -r requirements.txt

Execute the CIFAR-10 model:

	# CIFAR-10 pre-processing on dataset perform here.
	Execute below command:
	$ python3 pre_processing.py

	# Training the model
	Execute below command:
	$ python3 build_model.py

	# Evaluate the model
	Execute below command:
	$ python3 evaluate_model.py

	# To check inference on the model
	Execute below command:
	$ python3 inference.py

