"""
This script will be runned once to convert all .jpg images into numpy arrays;
this will make the process of loading the data when training faster
Transdformations over the data, like croping and resizing, will also be done at this script 
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
PATH2ARR = root / "data" / "arrays64"
PATH2IMG1 = root / "data" / "artwork-data-1" / "artwork-1"
PATH2IMG2 = root / "data" / "artwork-data-2" / "artwork-2"

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
		img = plt.imread(path2img)
		img = process(img, path2img)
		img_name = img_file[:-4] + ".npy"
		save_path =  PATH2ARR / img_name
		np.save(save_path, img)

	for j, img_file in enumerate(os.listdir(PATH2IMG2)):
		path2img = PATH2IMG2 / img_file
		img = plt.imread(path2img)
		img = process(img, path2img)
		img_name = img_file[:-4] + ".npy"
		save_path =  PATH2ARR / img_name
		np.save(save_path, img)

if __name__ == "__main__":
	if not len(os.listdir(PATH2ARR)): 
		main()
	else: 
		print("IMAGES ALREADY CONVERTED")
	