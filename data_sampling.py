"""
This script is to show sample images and to determine which dimentions are going to be used 
for the network.  
"""
__author__ = "Adrian Hernandez" 

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import pathlib 
from pathlib import Path 

root = Path(".")
PATH_TO_IMGS = root / "data_2" / "painting" 

def main():
	fig, ax = plt.subplots()
	img = plt.imread(PATH_TO_IMGS / "0026.jpg")
	print(img.shape)
	img = cv2.resize(img, (128,128))
	#  img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	print(img.shape)
	ax.imshow(img)
	plt.show()

if __name__ == "__main__":
	main()