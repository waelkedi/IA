import os
from PIL import Image
import numpy as np
import cv2
from resizeimage import resizeimage

path = "Images/001"
for img_name in os.listdir(path):
	img_path = os.path.join(path, img_name)
	img = cv2.imread(img_path)
	if not img is None :
		height = np.size(img, 0)
		width = np.size(img, 1)
		if height >600 or  width >900 :
			print("Trop gros !!! {}\n".format(img_name))
