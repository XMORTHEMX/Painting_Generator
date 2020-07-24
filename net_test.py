"""
This module is for testing the network that generates 
drawings
"""

__author__ = "Adrian Hernandez"

import torch 
import torchvision
from networks64 import Generator
import matplotlib.pyplot as plt
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_imgs = 64

PATH =  Path(".") / "models/top_artists.pth"
netG = Generator(100, 3, 64)
netG.load_state_dict(torch.load(PATH))
netG = netG.to(device)
noise = torch.randn(4, 100, 1, 1).to(device)

def main():
	with torch.no_grad():
		imgs = netG(noise)
		grid = torchvision.utils.make_grid(imgs)
		print(grid.shape)
		grid = grid.cpu()
		grid = grid.transpose(0, 1)
		grid = grid.transpose(1, 2)
		plt.imshow(grid)
		plt.show()


if __name__ == "__main__":
	main() 