"""
This is the module in which the arquitectures of the generator 
and discriminator networks used for this GAN are defined
"""
__author__ = "Adrian Hernadnez"

import torch 
import torch.nn as nn

class Generator(nn.Module):
	def __init__(self, noise_channels, out_channels, features):
		super(Generator, self).__init__()
		self.net = nn.Sequential(self.layer(in_channels = noise_channels, out_channels = features * 8, 
											padding = 0, stride = 1),
								 self.layer(in_channels = features * 8, out_channels = features * 4),
								 self.layer(in_channels = features * 4, out_channels = features * 2), 
								 self.layer(in_channels = features * 2, out_channels = features),
								 self.layer(in_channels = features, out_channels = out_channels, 
								 		    activation = False),
								 nn.Tanh())

	def layer(self, in_channels, out_channels, 
			  kernel_size = 4, stride = 2, padding = 1, 
			  batchnorm = True, activation = True):
		"""
		This function returns a conv-transposed-layer with an activation 
		function and batchnorm given the parameters 
		"""
		_layers = []
		_layers.append(nn.ConvTranspose2d(in_channels, out_channels, 
										  kernel_size = kernel_size, 	
										  stride = stride, padding = padding))
		if batchnorm:
			_layers.append(nn.BatchNorm2d(out_channels))
		if activation:
			_layers.append(nn.ReLU())
		
		return nn.Sequential(*_layers)
	
	def forward(self, x):
		x = self.net(x)
		return x

class Discriminator(nn.Module):
	def __init__(self, in_channels, features):
		super(Discriminator, self).__init__()
		self.net = nn.Sequential(self.layer(in_channels = in_channels, out_channels = features),
								 self.layer(in_channels = features, out_channels = features * 2),
								 self.layer(in_channels = features * 2, out_channels = features * 4),
								 self.layer(in_channels = features * 4, out_channels = features * 8),
								 self.layer(in_channels = features * 8, out_channels = 1, batchnorm = False,
								 			activation = False, stride = 1, padding = 0),
								 nn.Sigmoid())

	def layer(self, in_channels, out_channels, 
		      kernel_size = 4, stride = 2, padding = 1, 
		      batchnorm = True, activation = True):
		""" 
		This function returns a conv-layer with an activation and 
		batch normalization layer given the parameters passed
		"""
		_layers = []
		_layers.append(nn.Conv2d(in_channels, out_channels, 
					   			 kernel_size = kernel_size, 
					   			 stride = stride, padding = padding))
		if batchnorm:
			_layers.append(nn.BatchNorm2d(out_channels))
		if activation:
			_layers.append(nn.LeakyReLU(0.2))

		return nn.Sequential(*_layers)

	def forward(self, x):
		x = self.net(x)
		
		return x