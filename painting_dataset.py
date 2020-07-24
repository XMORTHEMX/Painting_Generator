"""
This is a module containing the dataset class of the paintings for the training of the GAN.
"""
__author__ = "Adrian Hernandez"

import torch 
from torch.utils.data import Dataset
import pathlib 
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np 

class PaintingDataset(Dataset):
	def __init__(self, annotations, root_dir, transforms = None):
		"""
		takes the path to annotations and to the root dir of the arrays, if 
		transforms are passed as an argument they will be applied. 
		"""

		self._path_to_annotations = Path(".") / annotations
		self.annotations = pd.read_csv(self._path_to_annotations)
		self.root_dir = Path(root_dir)
		self.transform = transforms

	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, indx):
		path_to_img = self.root_dir / self.annotations.iloc[indx, 0]
		img = np.load(path_to_img)
		label = torch.tensor(self.annotations.iloc[indx, 1])
		if self.transform:
			img = self.transform(img)

		return (img, label)