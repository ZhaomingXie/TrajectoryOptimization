import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

dtype = torch.FloatTensor

class StateConstraint(nn.Module):
	def __init__(self, input_size):
		super(StateConstraint, self).__init__()
		self.affine1 = nn.Linear(input_size, 16)
		self.affine2 = nn.Linear(16, 1)
		#for m in self.modules():
		#for m in self.modules():
		#	if isinstance(m, nn.Linear):
		#		m.weight.data.copy_(torch.zeros(m.weight.size()))
		#		m.bias.data.copy_(torch.zeros(m.bias.size()))
		#for name, p in self.named_parameters():
            # init parameters
		#	if 'bias' in name:
		#		p.data.fill_(0)
		#	if 'weight' in name:
		#		p.data.fill_(0)
		self.train()

	def forward(self, inputs):
		x = (inputs - 1) / 4
		#x = F.tanh(self.affine1(x))
		x = self.affine1(inputs)
		x = F.tanh(x)
		x = F.tanh(self.affine2(x))
		#output = F.tanh(self.affine2(x))
		return x

	def f(self, inputs):
		x = torch.from_numpy(inputs)
		x = Variable(x.type(torch.FloatTensor), requires_grad=True)
		return self(x).data.numpy()

	def f_x(self, inputs):
		x = torch.from_numpy(inputs)
		x = Variable(x.type(torch.FloatTensor), requires_grad=True)
		self.zero_grad()
		y = self(x)
		y.backward(retain_variables=True)
		return x.grad.data.numpy()