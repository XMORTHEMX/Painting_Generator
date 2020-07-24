"""
This module is runned once to create a csv file containing the path to the 
data arrays, this file will be used in the Dataset class
"""
__author__ = "Adrian Hernandez"

import pandas as pd
import csv
import os
import pathlib
from pathlib import Path 

root = Path(".")
PATH2ARR = root / "data_2" / "arrays_drawingsclean64"
PATH2ANN = root / "data_2" / "annotations_drawings_clean.csv"

def main():
	with open(PATH2ANN, mode = "w+", newline="") as file:
		writer = csv.writer(file)
		for arr in os.listdir(PATH2ARR):
			writer.writerow([arr, 1])

if __name__ == "__main__":
	if not PATH2ANN.is_file():
		main()
	else:
		print("ANNOTATIONS ALREADY CREATED")