# Form of the training and the test set:
# Input: an image.
# Output: 0 or 1. 0 means that the corresponding image is a woman and 1 means that it is a man.

import data_set as ds

from sklearn.model_selection import train_test_split
import os

def load(test_set_proportion=0.1, shuffle_data=True):
	"""
	Load the data set. This function generate a training set and a test set from the 'data_set' folder.
	Form of the training and the test set:
	Input: an image.
	Output: 0 or 1. 0 means that the corresponding image is a woman and 1 means that it is a man.

	:test_set_proportion: Proportion of test set.
	:shuffle_data: If True, the training set and the test set will be shuffled.
	:return: Two tuples. The first one has two components: training inputs and training outputs. The second one has
		also two components: test inputs and test outputs. Here is what is returned by the function: (x_train, y_train),
		(x_test, y_test).
	"""

	# Read the attributes.
	attributes = ds.read_attributes(include_gender=True)

	# Compute the file paths and the genders.
	file_paths = []
	genders = []
	for file_name, att in sorted(attributes.items()):
		file_paths.append(os.path.join(ds.TRAINING_IMAGE_PATH, file_name))

		if att[0] == 'male': # If male, then the value will be '1'. If female, then it will be '0'.
			gender = 1
		else:
			gender = 0
		genders.append(gender)

	# Read the images from the file paths.
	images = ds.read_images(file_paths)

	# Balance the data set.
	images, genders = ds.balance(images, genders)

	# Compute the training set and the test set.
	x_train, x_test, y_train, y_test = train_test_split(images, genders, test_size=test_set_proportion,
		shuffle=shuffle_data)

	return ((x_train, y_train), (x_test, y_test))

def load_test_set(dir_path):
	"""
	Load a test set. This function creates a test set from a directory path. This one must contain two subdirectories
	named: "male" and "female". The "male" subdirectory must contain all male images and the "female" subdirectory must
	contain all female images.
	Form of the test set:
	Input: an image.
	Output: 0 or 1. 0 means that the corresponding image is a woman and 1 means that it is a man.

	:dir_path: The path of the test set.
	:return: The file paths and the test set. It is a tuple having three components. The first one is the file paths.
		The second one is the test inputs. The third one is the test outputs. Here is what is returned by the function:
		(file_paths, x_test, y_test).
	"""

	# Compute the 'male' directory path and the 'female' directory path.
	male_file_path = os.path.join(dir_path, 'male')
	female_file_path = os.path.join(dir_path, 'female')

	# Compute the male file paths and the female file paths.
	exclude_hidden_files = lambda file_name: file_name[0] != '.'
	male_file_names = sorted(filter(exclude_hidden_files, os.listdir(male_file_path)))
	male_file_paths = [os.path.join(male_file_path, file_name) for file_name in male_file_names]
	female_file_names = sorted(filter(exclude_hidden_files, os.listdir(female_file_path)))
	female_file_paths = [os.path.join(female_file_path, file_name) for file_name in female_file_names]
	file_paths = male_file_paths + female_file_paths

	# Read the images (inputs).
	male_images = ds.read_images(male_file_paths)
	female_images = ds.read_images(female_file_paths)
	images = male_images + female_images

	# Compute the genders (outputs).
	genders = [1] * len(male_images) + [0] * len(female_images)

	return file_paths, images, genders

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	(x_train, y_train), (x_test, y_test) = load()

	print(y_train[:16])
	images = x_train[:16]
	for i, image in enumerate(images):
		plt.subplot(4, 4, (i + 1))
		plt.imshow(image)
	plt.show()
