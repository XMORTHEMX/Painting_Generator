
"""
In this module the neural net will be trained with paintings of top artist of the word
"""
__author__ = "Adrian Hernandez"

import torch
from torch.nn import BCELoss
import torch.optim as optim 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import networks64
from networks64 import Generator, Discriminator
import painting_dataset
from painting_dataset import PaintingDataset


PATH2MODEL = "./models/top_artists_2.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 0.0002
batch_size = 128
epochs = 300
img_size = 64
img_channels = 3
noise_channels = 256
features = 64
real_label = 1
fake_label = 0
fixed_noise = torch.randn(batch_size, noise_channels, 1, 1, device = device)

custom_transforms = transforms.Compose((transforms.ToTensor(),
										transforms.Normalize((0.5,),(0.5,))))
dataset = PaintingDataset(annotations = "data_3/annotations.csv", root_dir = "data_3/arrays64", 
						  transforms = custom_transforms)
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

netD = Discriminator(in_channels = img_channels, 
					 features = features).to(device)

netG = Generator(noise_channels = noise_channels, 
				 out_channels = img_channels, features = features).to(device)

optimD = optim.Adam(netD.parameters(), lr = lr, betas = (0.5, 0.999))
optimG = optim.Adam(netG.parameters(), lr = lr, betas = (0.5, 0.999))

netD.train()
netG.train()

criterion = BCELoss()

writer_real = SummaryWriter(f"top_artists_2/GAN_Paintings/real_images")
writer_fake = SummaryWriter(f"top_artists_2/GAN_Paintings/fake_images")


def train_D(fake_imgs, data, batch_size):
	netD.zero_grad()
	real_labels = (torch.ones(batch_size) * 0.9).to(device)
	fake_labels = (torch.ones(batch_size) * 0.1).to(device)
	predicts_real = netD(data)
	predicts_real = predicts_real.view(-1)
	lossD_real = criterion(predicts_real, real_labels)
	D_x = lossD_real.mean().item()
	predicts_fake = netD(fake_imgs.detach()).view(-1)
	lossD_fake = criterion(predicts_fake, fake_labels)
	lossD = lossD_fake + lossD_real
	lossD.backward()
	optimD.step()

	return lossD


def train_G(fake_imgs, batch_size):
	netG.zero_grad()
	labels = torch.ones(batch_size).to(device)
	predicts = netD(fake_imgs).view(-1)
	lossG = criterion(predicts, labels)
	lossG.backward()
	optimG.step()

	return lossG

def report(loss_G, loss_D, indx, epoch, data, step):
	print(f"[EPOCH {epoch}/{epochs}] \n Loss Discriminator: {loss_D} \n Loss Gnereator: {loss_G}")
	with torch.no_grad():
		fake = netG(fixed_noise)
		img_grid_real = torchvision.utils.make_grid(data[:48], normalize = True)
		img_grid_fake = torchvision.utils.make_grid(fake[:48], normalize = True)
		writer_real.add_image("Real_Images", img_grid_real, global_step = step)
		writer_real.add_image("Fake_Images", img_grid_fake, global_step = step)

			
def save():
	weights = netG.state_dict()
	torch.save(weights, PATH2MODEL)


def main():
	step = 0
	for epoch in range(epochs):
		for indx, (data, labels) in enumerate(dataloader):
			data = data.to(device)
			batch_size = data.shape[0]
			noise = torch.randn(batch_size, noise_channels, 1, 1).to(device)
			fake_imgs = netG(noise)
			loss_D = train_D(fake_imgs = fake_imgs, data = data, 
							 batch_size = batch_size)
			loss_G = train_G(fake_imgs = fake_imgs, batch_size = batch_size)
			
			if indx % 8 == 0:
				report(loss_G, loss_D, indx, epoch, data, step)
				step += 1
	save()

if __name__ == "__main__":
 	main()
	