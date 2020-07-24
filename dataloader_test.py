"""
This module is for testing the proper functioning of the dataloader
"""
__author__ = "Adrian Hernandez"

import matplotlib 
import matplotlib.pyplot as plt 
import torch 
from torch.utils.data import DataLoader 
import torchvision 
from torchvision import transforms 
from painting_dataset import PaintingDataset

my_transforms = transforms.Compose((transforms.ToTensor(),
									transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))))


data = PaintingDataset(annotations = "data/annotations.csv", 
					   root_dir = "data/arrays", 
					   transforms = my_transforms)

dataloader = DataLoader(dataset = data, batch_size = 128, shuffle = True)


def main():
	fig, ax = plt.subplots()
	for data, label in dataloader:
		print(data.shape)
		grid = torchvision.utils.make_grid(data)
		print(f"shape of the grid {grid.shape}")
		grid = grid.transpose(0,1)
		grid = grid.transpose(1,2)
		ax.imshow(grid[:640,:640]) 
		break

	plt.show()
if __name__ == "__main__":
	main()
