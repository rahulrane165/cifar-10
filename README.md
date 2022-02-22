# cifar-10

## A Keras-TensorFlow Convolutional Neural Network used for training and testing on the CIFAR-10 dataset.

CIFAR10 dataset includes:

1. 50,000 32x32 color training images.
2. 10,000 32x32 color testing images.
3. 10 different classes.
4. 6,000 images per class.


### Prerequisites:
#### Hardware Requirements:
	Development PC (OS:Ubuntu 18.04.6 LT)

### Installations:
#### Please follow below commands.
	$ pip3 install --upgrade pip3 setuptools
	$ pip3 install -r requirements.txt


### Step 1:
#### Clone the git package:
#### Open terminal on Ubuntu PC:
	$ git clone --recursive https://github.com/rahulrane165/cifar-10
	$ cd cifar-10

### Step 2:
#### Dataset pre-processing:
	1. Go to https://www.cs.toronto.edu/~kriz/cifar.html
	2. Select CIFAR-10 python version and download dataset which is 163MB.
	3. Copy cifar-10-python.tar.gz to Ubuntu PC to current directory.
	4. Open terminal on Ubuntu PC:
		$ tar -xvf cifar-10-python.tar.gz
	5. cifar-10-batches-py package contains following data:
		1. test_batch
		2. batches.meta
		3. data_batch_1
		4. data_batch_2
		5. data_batch_3
		6. data_batch_4
		7. data_batch_5
	Python script help to unpickle dataset and load into appropriate training format.
	6. $ cd cifar-10/
	7. $ python3 pre_processing.py

### Step 3:
#### Train custom design CNN model on CIFAR-10 dataset:
	$ cd cifar-10/
	$ python3 build_model.py
	Output model: saved_models/cifar-10_model.h5
	Loss plot: loss_plot.png
	Accuracy plot: accuracy_plot.png
	NOTE: Hyper-parameters can be tune in script.
	Current hyper-parameters set to:
	batch_size = 32
	epochs = 200

### Setp 4:
#### Evaluate trained model on test dataset:
	$ cd cifar-10/
	$ python3 evaluate_model.py
	Current evaluation model accuracy is: 88.740
	Output: confusion_matrix_plot.png

### Step 5:
#### Run inference on trained model:
##### Copy data to Input folder and change path in script.
	$ cd cifar-10/
	$ python3 python3 inference.py
