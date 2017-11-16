from optimal_control import SQP
from constraint_network import StateConstraint

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import random

import numpy        as np
import scipy.sparse as sp
from optimize.snopt7 import SNOPT_solver
import matplotlib.pyplot as plt
from utils import PointMassTransitionFunction, QuadraticCostFunction, QuadraticCostFinalFunction, CircleConstraintFunction


point_dynamics = PointMassTransitionFunction(0.1)
quadratic_cost = QuadraticCostFunction(0 * np.identity(4), 1*np.identity(2))
quadratic_final_cost = QuadraticCostFinalFunction(100*np.identity(4))
t = SQP(np.array([2.0, 2.0, 0.0, 0.0]), 2, 100, c=quadratic_cost, c_f=quadratic_final_cost, f=point_dynamics)
#t.add_state_constraint(CircleConstraintFunction(np.ones(2), 0.5))
#t.set_conctrol_constraint(0.2 * np.ones(2))
#states, control = t.optimize()

test_constraint = StateConstraint(4)
circle_constraint = CircleConstraintFunction(np.ones(2), 0.5)
sucess = False

training_set = []

optimizer = optim.Adam(test_constraint.parameters(), lr=0.001)
test_constraint.train()
avg_loss = 0
t.state_constraint = []
t.add_state_constraint(test_constraint)
solution, states, control = t.optimize(guess=True, initial_guess=np.zeros(t.num_variable))
while not sucess:
	valid = np.zeros((4, 100*100))
	index = 0
	for i in range(100):
		for j in range(100):
			x = -1 + 0.04 * i
			y = -1 + 0.04 * j
			point = np.zeros(4)
			point[0] = x
			point[1] = y
			point = torch.from_numpy(point)
			point = Variable(point.type(torch.FloatTensor))
			if test_constraint(point).data[0] <= 0:
				valid[0, index] = x
				valid[1, index] = y
				index += 1
	fig,ax = plt.subplots()
	ax.scatter(valid[0,:],valid[1,:])
	#circle1 = plt.Circle((1, 1), 0.5, color='r')
	#ax.add_artist(circle1)
	plt.axis('equal')
	plt.axis([-1, 3, -1, 3])
	plt.show()
	programPause = raw_input("Press the <ENTER> key to continue...")
	plt.close(fig)
	plt.clf()

	success_number = 0
	data_collect = []
	target_collect = []
	states = np.zeros((4, 10000))
	for i in range(100):
		for j in range(100):
			states[0, i * 100 + j] = -1 + 0.04 * i
			states[1, i * 100 + j] = -1 + 0.04 * j 
	for i in range(100 * 100):
		if circle_constraint.f(states[:, i]) <= 0:
			data = torch.from_numpy(states[:, i]).unsqueeze(0)
			target = -torch.ones(1, 1)
			data, target = Variable(data.type(torch.FloatTensor), requires_grad=True), Variable(target.type(torch.FloatTensor), requires_grad=True)
			#training_set.append([data, target])
			data_collect.append(data)
			target_collect.append(target)
			success_number += 1
		else:
			data = torch.from_numpy(states[:, i]).unsqueeze(0)
			target = torch.ones(1, 1)
			data, target = Variable(data.type(torch.FloatTensor), requires_grad=True), Variable(target.type(torch.FloatTensor), requires_grad=True)
			data_collect.append(data)
			target_collect.append(target)
			#training_set.append([data, target])
	for event in zip(*[data_collect, target_collect]):
		training_set.append(event)
	if success_number < -1:
		break
	print "success number", success_number
	print "avg_loss", avg_loss
	print "checking"
	check = np.zeros(4)
	print test_constraint.f_x(check)
	solution, states, control = t.optimize(guess=True, initial_guess=solution)
	fig,ax = plt.subplots()
	ax.scatter(states[0,:],states[1,:])
	circle1 = plt.Circle((1, 1), 0.5, color='r')
	ax.add_artist(circle1)
	plt.axis('equal')
	plt.axis([-1, 3, -1, 3])
	plt.show()
	programPause = raw_input("Press the <ENTER> key to continue...")
	plt.close(fig)
	plt.clf()
	for i in range(10000):
		samples = zip(*random.sample(training_set, 128))
		batch_data, batch_target = map(lambda x: torch.cat(x, 0), samples)
		batch_output = test_constraint(batch_data)
		batch_output = batch_output.squeeze(0)
		loss = torch.mean((batch_output - batch_target)**2)
		#print "batch_target", batch_output
		optimizer.zero_grad()
		loss.backward(retain_variables=True)
		optimizer.step()
		print loss
		#programPause = raw_input("Press the <ENTER> key to continue...")
	print "loss", torch.mean((batch_output - batch_target)**2)
	print list(test_constraint.parameters())[0].data
	print list(test_constraint.parameters())[1].data
	print list(test_constraint.parameters())[2].data
	print list(test_constraint.parameters())[3].data
	print batch_output.size(), batch_target.size()
	programPause = raw_input("Press the <ENTER> key to continue...")
	t.state_constraint = []
	t.add_state_constraint(test_constraint)
	solution, states, control = t.optimize(guess=True, initial_guess=solution)

fig,ax = plt.subplots()
ax.scatter(states[0,:],states[1,:])
circle1 = plt.Circle((1, 1), 0.5, color='r')
ax.add_artist(circle1)
plt.axis('equal')
plt.axis([-1, 3, -1, 3])
plt.show()