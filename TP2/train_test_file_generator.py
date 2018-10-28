# Create the 'train.txt' and test.txt' files.

import os

data_path = './images'
train_file_path = 'train.txt'
test_file_path = 'test.txt'
num_test_data = 250
# Path of the folder that contains all iamges in Google Colab.
#google_colab_image_path = '/content/gdrive/My\\ Drive/defi2/darknet/keys_detection/images/'
google_colab_image_path = './images/'

# Get the images names.
image_names = []
for file_name in sorted(os.listdir(data_path)):
	if file_name[0] == '.': # Skip this iteration if the file is hidden.
		continue

	# Get the extension of the file.
	extension = os.path.splitext(file_name)[1]

	# Skip if the file is a text file.
	if extension == '.txt':
		continue

	image_names.append(file_name) # 'file_name' corresponds to an image.

# Compute the train and test iamge names.
train_image_names = image_names[num_test_data:]
test_image_names = image_names[:num_test_data]
print(len(train_image_names))
print(len(test_image_names))

# Compute the train and test image paths.
train_image_paths = [os.path.join(google_colab_image_path, image_name) for image_name in train_image_names]
test_image_paths = [os.path.join(google_colab_image_path, image_name) for image_name in test_image_names]

# Write the train and the test file.
with open(train_file_path, 'w') as file:
	file.write('\n'.join(train_image_paths))
	file.write('\n')

with open(test_file_path, 'w') as file:
	file.write('\n'.join(test_image_paths))
	file.write('\n')
