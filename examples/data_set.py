import csv
import cv2
import numpy as np
import math
import collections
import os

# Important paths.
# Path of the folder that contains the training images and the attributes.
DATA_SET_PATH = os.path.join(os.pardir, 'data_set')
# Path of the training set.
TRAINING_SET_PATH = os.path.join(DATA_SET_PATH, 'training')
# Path of the folder that containts the training images.
TRAINING_IMAGE_PATH = os.path.join(TRAINING_SET_PATH, 'images')
# Path of the file containing the attributes of the training set.
TRAINING_ATTRIBUTE_FILE_PATH = os.path.join(TRAINING_SET_PATH, 'attributes.csv')
# Path of the test set.
TEST_SET_PATH = os.path.join(DATA_SET_PATH, 'test')

# Image dimensions.
IMAGE_WIDTH = 250
IMAGE_HEIGHT = 250
IMAGE_COLOR_DIMENSION = 3

# Default parameter for the iamge transformation.
DEFAULT_BRIGHTNESS_RANGE = (-8, 8)
DEFAULT_CONTRAST_RANGE = (-8, 8)
DEFAULT_ROTATION_DEGREE_RANGE = (-10, 10)
DEFAULT_ZOOM_LEVEL_RANGE = (0.9, 1.2)

def read_attributes(include_gender=False, include_age=False, include_smile=False, include_moustache=False,
	include_beard=False, include_sideburns=False, include_glasses=False, include_emotion=False, include_bald=False,
	include_visible_hair=False, include_hair_colour=False, include_eye_makeup=False, include_lip_makeup=False,
	include_all_attributes=False):
	"""
	Read the attributes.

	:include_gender: If True, then the gender will be included in the output.
	:include_age: If True, then the age will be included in the output.
	:include_smile: If True, then the information about the smile will be included in the output.
	:include_moustache: If True, then the information about the moustache will be included in the output.
	:include_beard: If True, then the information about the beard will be included in the output.
	:include_sideburns: If True, then the information about sideburns will be included in the output.
	:include_glasses: If True, then the information about glasses will be included in the output.
	:include_emotion: If True, then the information about the emotion will be included in the output.
	:include_bald: If True, then the information about the balness will be included in the output.
	:include_visible_hair: If True, then the information about the hair visibility will be included in the output.
	:include_hair_color: If True, then the information about the hair color will be included in the output.
	:include_eye_makeup: If True, then the information about the eye makeup will be included in the output.
	:include_lip_makeup: If True, then the information about the lip makeup will be included in the output.
	:include_all_attributes: If True, then all attributes will be included in the output (all other parameters will
		therefore be overrided). If False, some attriubtes will be included, depending on the values of the other
		parameters.
	:return: A dictionary. Each key is a file name (the path of a file can therefore be obtained as follows:
		'os.path.join(TRAINING_IMAGE_PATH, file_name)', where 'file_name' is the name of a file). Each value is a list
		of attributes (see the 'attribute_generator.py' file for more details about the attributes).
	"""

	if include_all_attributes:
		include_gender = include_age = include_smile = include_moustache = include_beard = include_sideburns = \
		include_glasses = include_emotion = include_bald = include_visible_hair = include_hair_colour = \
		include_eye_makeup = include_lip_makeup = True
	
	attributes = {}
	with open(TRAINING_ATTRIBUTE_FILE_PATH, newline='') as file:
		csv_reader = csv.reader(file, delimiter=';', quotechar='\"')
		next(csv_reader) # Skip the header.

		for row in csv_reader:
			# Key of the entry.
			file_name = row[0]

			# Value of the entry.
			att = []
			if include_gender:
				att.append(row[1])
			if include_age:
				att.append(int(row[2]))
			if include_smile:
				att.append(int(row[3]))
			if include_moustache:
				att.append(int(row[4]))
			if include_beard:
				att.append(int(row[5]))
			if include_sideburns:
				att.append(int(row[6]))
			if include_glasses:
				att.append(int(row[7]))
			if include_emotion:
				att.append(row[8])
			if include_bald:
				att.append(int(row[9]))
			if include_visible_hair:
				att.append(int(row[10]))
			if include_hair_colour:
				att.append(row[11])
			if include_eye_makeup:
				att.append(int(row[12]))
			if include_lip_makeup:
				att.append(int(row[13]))

			attributes[file_name] = att

	return attributes

def read_images(file_paths):
	"""
	Read images of the data set from file paths.

	:file_paths: A list of file paths.
	:return: A list of images in RGB format.
	"""

	images = []
	for file_path in file_paths:
		if os.path.isfile(file_path):
			image = cv2.imread(file_path, cv2.IMREAD_COLOR) # Read a image as BGR.
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB.

			images.append(image)

	return images

def balance(x, y, brightness_range=DEFAULT_BRIGHTNESS_RANGE, contrast_range=DEFAULT_CONTRAST_RANGE, allowed_flip=True,
	rotation_degree_range=DEFAULT_ROTATION_DEGREE_RANGE, zoom_level_range=DEFAULT_ZOOM_LEVEL_RANGE):
	"""
	Balance the data set for a single attribute by increasing the number of data.
	For example, suppose that the output vector ('y') contains 24 'val1', 60 'val2' and 16 'val3' ('val1', 'val2' and
	'val' can be a list, a string, etc.). This data set is not balanced. This function will increase the number of data
	with the 'val1' attribute and the 'val3' attribute to have 60 'val1', 60 'val2' and 60 'val3'.
	The increase of the dataset is done by slightly transforming the inputs (images):
	- increasing or decreasing the brightness and the contrast;
	- flip the image horizontally;
	- rotate the image;
	- zoom or dezoom on the image.
	Note: you should NOT use this function on continuous attributes.

	:x: The inputs (images). It is a list of Numpy arrays. Each element in the list is an instance. Each instance
		(Numpy array) have three dimensions. The first one is the image height. The second one is the image width. And
		the last one is the color dimension (IMAGE_COLOR_DIMENSION).
	:y: The outputs (int, string, list, etc.).
	:brightness_range: The brightness range. It is a tuple of two values. The first component is the lower bound and
		the second one is the upper bound. To disable this transformation, set this parameter to (0, 0).
	:contrast_range: The contrast range. It is a tuple of two values. The first component is the lower bound and the
		second one is the upper bound. To disable this transformation, set this parameter to (0, 0).
	:allowed_flip: If True, the image can be horizontally flipped.
	:rotation_degree_range: The rotation degree range. It is a tuple of two values. The first component is the lower
		bound and the second one is the upper bound. To disable this transformation, set this parameter to (0, 0).
	:zoom_level_range: The zoom level range. It is a tuple of two values. The first component is the lower bound and
		the second one is the upper bound. To disable this transformation, set this parameter to (1.0, 1.0).
	:return: The new inputs and the new outpus. Type: Numpy array.
	"""

	# Makes the outputs hashable if they are iterable.
	if len(y) > 0 and isinstance(y[0], collections.Iterable):
		y = [tuple(y_t) for y_t in y]

	# For each attribute, compute the number of data that have this attribute.
	num_data = {}
	for att in y:
		try: # If 'att' is already in the dictionary.
			num_data[att] += 1
		except KeyError: # If 'att' is not in the dictionary yet.
			num_data[att] = 1

	# Compute the number of data of the most frequent attribute.
	# The goal is that, for each attribute, there are 'max_num_data' data in the new data set.
	max_num_data = max(num_data.values())

	# Compute the factor to reach 'max_num_data'.
	# For example, suppose that	'max_num_data' is equal to 10 and 'num_data[att1]' is equal to 3. Then, the factor for
	# this attribute ('factor_to_reach_max_value[att1]') is equal to 4. This means that it is necessary to multiply by
	# 4 the number of data that have this attribute.
	factor_to_reach_max_value = {
		att: math.ceil(max_num_data / num_dt)
		for att, num_dt in num_data.items()
	}

	# Increase the number of data.
	new_x = list(x) # Copy and convert to list.
	new_y = list(y) # Copy and convert to list.
	# For each attribute, 'num_new_data' contains the number of new data that have this attribute.
	num_new_data = num_data
	for image, att in zip(x, y):
		for i in range(factor_to_reach_max_value[att]):
			if num_new_data[att] < max_num_data:
				# Create a new image by transforming the old one.
				new_image = transform_image(image)

				# Add the new image to the new data set.
				new_x.append(new_image)
				new_y.append(att)

				# Increment by 1 the counter of this attribute.
				num_new_data[att] += 1
			else:
				break

	return np.array(new_x), np.array(new_y)

def create_data(x, y, num_data, brightness_range=DEFAULT_BRIGHTNESS_RANGE, contrast_range=DEFAULT_CONTRAST_RANGE,
	allowed_flip=True, rotation_degree_range=DEFAULT_ROTATION_DEGREE_RANGE, zoom_level_range=DEFAULT_ZOOM_LEVEL_RANGE):
	"""
	Create new data from old one.
	The creation of data is done by slightly transforming some inputs (images):
	- increasing or decreasing the brightness and the contrast;
	- flip the image horizontally;
	- rotate the image;
	- zoom or dezoom on the image.

	:x: The inputs (images). It is a list of Numpy arrays. Each element in the list is an instance. Each instance
		(Numpy array) have three dimensions. The first one is the image height. The second one is the image width. And
		the last one is the color dimension (IMAGE_COLOR_DIMENSION).
	:y: The outputs.
	:num_data: Number of data to generate.
	:brightness_range: The brightness range. It is a tuple of two values. The first component is the lower bound and
		the second one is the upper bound. To disable this transformation, set this parameter to (0, 0).
	:contrast_range: The contrast range. It is a tuple of two values. The first component is the lower bound and the
		second one is the upper bound. To disable this transformation, set this parameter to (0, 0).
	:allowed_flip: If True, the image can be horizontally flipped.
	:rotation_degree_range: The rotation degree range. It is a tuple of two values. The first component is the lower
		bound and the second one is the upper bound. To disable this transformation, set this parameter to (0, 0).
	:zoom_level_range: The zoom level range. It is a tuple of two values. The first component is the lower bound and
		the second one is the upper bound. To disable this transformation, set this parameter to (1.0, 1.0).
	:return: The inputs and outputs created. Type: Numpy array.
	"""

	new_x = []
	new_y = []
	data_set_size = len(x)
	for i in range(num_data):
		i_data = np.random.randint(data_set_size)
		image = x[i_data]
		att = y[i_data]

		new_x.append(transform_image(image))
		new_y.append(att)

	return np.array(new_x), np.array(new_y)

def transform_image(image, brightness_range=DEFAULT_BRIGHTNESS_RANGE, contrast_range=DEFAULT_CONTRAST_RANGE,
	allowed_flip=True, rotation_degree_range=DEFAULT_ROTATION_DEGREE_RANGE, zoom_level_range=DEFAULT_ZOOM_LEVEL_RANGE):
	"""
	Randomly transforms an image.
	Increase or decrease the brightness and the contrast. Filp or not the image horizontally. Rotate the image. Zoom or
	dezoom on the image.

	:image: The image.
	:brightness_range: The brightness range. It is a tuple of two values. The first component is the lower bound and
		the second one is the upper bound. To disable this transformation, set this parameter to (0, 0).
	:contrast_range: The contrast range. It is a tuple of two values. The first component is the lower bound and the
		second one is the upper bound. To disable this transformation, set this parameter to (0, 0).
	:allowed_flip: If True, the image can be horizontally flipped.
	:rotation_degree_range: The rotation degree range. It is a tuple of two values. The first component is the lower
		bound and the second one is the upper bound. To disable this transformation, set this parameter to (0, 0).
	:zoom_level_range: The zoom level range. It is a tuple of two values. The first component is the lower bound and
		the second one is the upper bound. To disable this transformation, set this parameter to (1.0, 1.0).
	:return: The transformed image.
	"""

	# Change the brightness and the contrast.
	brightness = np.random.randint(brightness_range[0], brightness_range[1] + 1) # Random brightness.
	contrast = np.random.randint(contrast_range[0], contrast_range[1] + 1) # Random contrast.
	image = cv2.addWeighted(image, 1. + (contrast / 127.), image, 0, brightness - contrast)

	# Flip the image horizontally.
	if allowed_flip:
		flip_image = np.random.choice([True, False]) # Random boolean (either true or false).
		if flip_image:
			image = cv2.flip(image, 1)

	# Rotate the image.
	rotation_degree = np.random.randint(rotation_degree_range[0], rotation_degree_range[1] + 1) # Random degree between.
	zoom_level = np.random.uniform(zoom_level_range[0], zoom_level_range[1]) # Random zoom level.
	width, height, _ = image.shape
	rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_degree, zoom_level)
	image = cv2.warpAffine(image, rotation_matrix, (width, height))
	
	return image

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	attributes = read_attributes(include_all_attributes=True)
	for i, (key, value) in enumerate(attributes.items()):
		print("{}: {}".format(key, value))
		if i == 9:
			break
	print()

	#####
	
	file_names = ['Aaron_Eckhart_0001.jpg', 'Aaron_Guiel_0001.jpg', 'Aaron_Patterson_0001.jpg', 'Aaron_Peirsol_0001.jpg',
		'Aaron_Peirsol_0002.jpg', 'Aaron_Peirsol_0003.jpg']
	file_paths = [os.path.join(TRAINING_IMAGE_PATH, file_name) for file_name in file_names]

	attributes = read_attributes(include_gender=True, include_smile=True, include_hair_colour=True)
	attributes = [attributes[file_name] for file_name in file_names] # Select some images.
	images = read_images(file_paths)
	for file_name, att in zip(file_names, attributes):
		print("{}: {}".format(file_name, att))

	new_images, new_attributes = balance(images, attributes)
	for att in new_attributes:
		print("{}".format(att))

	for i, image in enumerate(new_images):
		plt.subplot(4, 4, (i + 1))
		plt.imshow(image)
	plt.show()

	"""
	(x_train, y_train), (x_test, y_test) = load()

	print(y_train[:16])
	images = x_train[:16]
	for i, image in enumerate(images):
		plt.subplot(4, 4, (i + 1))
		plt.imshow(image)
	plt.show()
	"""
