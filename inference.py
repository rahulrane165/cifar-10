# Python scipt to execute inference on CIFAR-10 model

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(32, 32))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 32, 32, 3)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

if __name__ == "__main__":
	"""Main function to run inference"""

	class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	img = load_image('Input/airplane.jpg')

	#Load model
	model = load_model('saved_models/cifar-10_model.h5')

	#Run inference on image
	result = model.predict_classes(img)
	print("Model predict :", class_labels[result[0]])

