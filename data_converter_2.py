"""
This module preprocess the painting only data for training_2
"""
__author__ = "Adrian Hernandez"

import numpy as np 
import os
import pathlib 
from pathlib import Path 
from matplotlib import pyplot as plt 
import cv2 


IMGSIZE = 64
root = Path(".")
PATH2ARR = root / "data_2" / "arrays_drawingsclean64"
PATH2IMG1 = root / "data_2" / "drawings"


def process(img, path):
	"""
	This function resizes the images to the dimensions that the neural net will have as inputs.
	There are some images in the dataset that are grayscale so for thoose cases they will be converted 
	to a BGR image. 
	"""
	img = cv2.resize(img,(IMGSIZE, IMGSIZE))
	if len(img.shape) != 3:
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	return img


def main():
	for i, img_file in enumerate(os.listdir(PATH2IMG1)):
		path2img = PATH2IMG1 / img_file
		try:
			img = plt.imread(path2img)
		except Exception as e:
			print(path2img, "unreadable")
			continue
		img = process(img, path2img)
		img_name = str(i) + ".npy"
		save_path =  PATH2ARR / img_name
		np.save(save_path, img)



if __name__ == "__main__":
	if not len(os.listdir(PATH2ARR)): 
		main()
	else: 
		print("IMAGES ALREADY CONVERTED")