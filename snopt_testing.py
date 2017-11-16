import numpy        as np
import scipy.sparse as sp
from optimize.snopt7 import SNOPT_solver
import matplotlib.pyplot as plt
from utils import PointMassTransitionFunction, QuadraticCostFunction, QuadraticCostFinalFunction, CircleConstraintFunction

#simple trajectory optimization solver using SNOPT
#minimize \sum c(xt, ut) + c_f(x_f, u_f)
#subject to x_{t+1} = f(x_t, u_t)

class TrajectoryOptimization():
	#initilization, parameters includes:
	#initial state x_0
	#time horizon T
	#cost function c
	#final cost function c_f
	#transition function f
	def __init__(self, x_0, num_control, T, c=None, c_f=None, f=None):
		self.x_0 = x_0
		self.T = T
		self.c = c
		self.c_f = c_f
		self.f = f
		self.num_control = num_control
		self.num_state = x_0.shape[0]
		self.state_constraints = []
		self.control_constraint = 1e10 * np.ones(num_control) #assume box control contraints
		#print "num_state", self.num_state

	def set_cost(self, c):
		self.c = c

	def set_final_cost(self, c_f):
		self.c_f = c_f

	def set_f(self, f):
		self.f = f

	def add_state_constraint(self, constraint):
		self.state_constraints.append(constraint)

	def set_conctrol_constraint(self, control_constraint):
		self.control_constraint = control_constraint

	def optimize(self):
		def sntoya_objF(status,x,needF,needG,cu,iu,ru):
			F = np.zeros(1 + self.num_state * self.T + len(self.state_constraints) * (self.T + 1)) #objective row + transition constraint
			states = np.zeros((self.num_state, self.T + 1)) #T + 1 states
			control = np.zeros((self.num_control, self.T)) #T control
			for i in range(self.T):
				states[:, i] = x[i * (self.num_state + self.num_control):i * (self.num_state + self.num_control) + self.num_state]
				control[:, i] = x[i * (self.num_state + self.num_control) + self.num_state:(i+1) * (self.num_state + self.num_control)]
			states[:, self.T] = x[self.T * (self.num_state + self.num_control):self.T * (self.num_state + self.num_control) + self.num_state]
			
			#compute objective
			index = 0
			for i in range(self.T):
				F[index] += self.c.f(states[:, i], control[:, i])
			F[index] += self.c_f.f(states[:, self.T])
			index = index + 1

			#compute transition constraints
			for i in range(self.T):
				F[index:index+self.num_state] = states[:, i + 1] - self.f.f(states[:, i], control[:, i])
				index = index + self.num_state

			#compute state constraints
			for i in range(len(self.state_constraints)):
				for j in range(self.T + 1):
					state_constraint = self.state_constraints[i]
					F[index] = state_constraint.f(states[:, j])
					index += 1
			return status, F

		def sntoya_objFG(status,x,needF,needG,cu,iu,ru):
			F = np.zeros(1 + self.num_state * self.T + len(self.state_constraints) * (self.T + 1)) #objective row + transition constraint
			states = np.zeros((self.num_state, self.T + 1)) #T + 1 states
			control = np.zeros((self.num_control, self.T)) #T control
			for i in range(self.T):
				states[:, i] = x[i * (self.num_state + self.num_control):i * (self.num_state + self.num_control) + self.num_state]
				control[:, i] = x[i * (self.num_state + self.num_control) + self.num_state:(i+1) * (self.num_state + self.num_control)]
			states[:, self.T] = x[self.T * (self.num_state + self.num_control):self.T * (self.num_state + self.num_control) + self.num_state]
			
			#compute objective
			index = 0
			for i in range(self.T):
				F[index] += self.c.f(states[:, i], control[:, i])
			F[index] += self.c_f.f(states[:, self.T])
			index = index + 1

			#compute transtition constraints
			for i in range(self.T):
				F[index:index+self.num_state] = states[:, i + 1] - self.f.f(states[:, i], control[:, i])
				index = index + self.num_state

			#compute state constraints
			for i in range(len(self.state_constraints)):
				for j in range(self.T + 1):
					state_constraint = self.state_constraints[i]
					F[index] = state_constraint.f(states[:, j])
					index += 1

			#fill in gradient information
			num_variable = self.T * (self.num_state + self.num_control) + self.num_state
			#objective gradients: invlove all variable
			#transition constraint gradients: invlove current state + current control + next state
			#state constraint gradients: involve current state
			G = np.zeros(num_variable + self.num_state * self.T * (self.num_state * 2 + self.num_control) + len(self.state_constraints) * self.num_state * (self.T + 1))
			index = 0

			#fill in objective gradients
			for i in range(self.T):
				G[index:index+self.num_state] = self.c.f_x(states[:, i], control[:, i])
				index += self.num_state
				G[index:index+self.num_control] = self.c.f_u(states[:, i], control[:, i])
				index += self.num_control
			G[index:index+self.num_state] = self.c_f.f_x(states[:, self.T])
			index += self.num_state

			#fill in transition constraint gradients
			for i in range(self.T):
				f_x = -self.f.f_x(states[:, i], control[:, i])
				f_u = -self.f.f_u(states[:, i], control[:, i])
				f_x_next = np.identity(self.num_state)
				for j in range(self.num_state):
					G[index:index+self.num_state] = f_x[j, :]
					index += self.num_state
					G[index:index+self.num_control] = f_u[j, :]
					index += self.num_control
					G[index:index+self.num_state] = f_x_next[j, :]
					index += self.num_state

			#fill in state constraints gradients
			for i in range(len(self.state_constraints)):
				for j in range(self.T + 1):
					state_constraint = self.state_constraints[i]
					G[index:index+self.num_state] = state_constraint.f_x(states[:, j])
					index += self.num_state
			return status, F, G

		inf = 1.0e20
		snopt = SNOPT_solver()
		snopt.setOption('Verbose',True)
		snopt.setOption('Solution print',True)
		snopt.setOption('Print file','sntoya_testing.out')
		#xNames  = np.array([ '      x0', '      x1' ])
		#FNames  = np.array([ '      F0', '      F1', '      F2' ],dtype='c')
		num_variable = self.T * (self.num_state + self.num_control) + self.num_state
		num_F = 1 + self.num_state * self.T + len(self.state_constraints) * (self.T + 1)
		x0      = np.zeros(num_variable)
		xlow    = np.ones(num_variable) * -inf
		xupp    = np.ones(num_variable) * inf
		Flow    = np.zeros(num_F)
		Fupp    = np.zeros(num_F)
		Flow[0] = -inf
		Fupp[0] = inf
		Flow[1 + self.num_state * self.T: 1 + self.num_state * self.T + len(self.state_constraints) * (self.T + 1)] = -inf * np.ones(len(self.state_constraints) * (self.T + 1))
		xlow[0:self.num_state] = self.x_0
		xupp[0:self.num_state] = self.x_0
		ObjRow  = 1

		#set control constraint
		for i in range(self.T):
			xlow[i * (self.num_state + self.num_control) + self.num_state: (i+1) * (self.num_state + self.num_control)] = -self.control_constraint
			xupp[i * (self.num_state + self.num_control) + self.num_state: (i+1) * (self.num_state + self.num_control)] = self.control_constraint

		#construct gradient matrix
		A = np.zeros((num_F, num_variable))
		G = np.zeros((num_F, num_variable))

		#fill in gradient entry with 1s
		G[0, :] = np.ones(num_variable) #objective row
		#fill in transition constraints entry
		for i in range(self.T):
			start_index = i * (self.num_state + self.num_control)
			end_index = (i + 1) * (self.num_state + self.num_control) + self.num_state
			G[1 + i * self.num_state: 1 + (i+1)*self.num_state, start_index:end_index] = np.ones((self.num_state, self.num_state * 2 + self.num_control))
		#fill in state constraints entry
		index = 1 + self.num_state * self.T
		for i in range(len(self.state_constraints)):
			for j in range(self.T  + 1):
				start_index = j * (self.num_state + self.num_control)
				end_index = start_index + self.num_state
				G[index, start_index:end_index] = np.ones(self.num_state)
				index += 1

		#A = sp.coo_matrix(A)
		#G = sp.coo_matrix(G)
		#snopt.setOption('Verify level',3)

		snopt.snopta(name='sntoyaFG',usrfun=sntoya_objFG,x0=x0,xlow=xlow,xupp=xupp, Flow=Flow,Fupp=Fupp,ObjRow=ObjRow, G=G)
		states = np.zeros((self.num_state, self.T + 1))
		control = np.zeros((self.num_control, self.T))
		for i in range(self.T):
			states[:, i] = snopt.x[i * (self.num_state + self.num_control):i * (self.num_state + self.num_control) + self.num_state]
			control[:, i] = snopt.x[i * (self.num_state + self.num_control) + self.num_state:(i+1) * (self.num_state + self.num_control)]
		states[:, self.T] = snopt.x[self.T * (self.num_state + self.num_control):self.T * (self.num_state + self.num_control) + self.num_state]
		return states, control

class iLQR():
	def __init__():
		def __init__(self, x_0, num_control, T, c=None, c_f=None, f=None):
		self.x_0 = x_0
		self.T = T
		self.c = c
		self.c_f = c_f
		self.f = f
		self.num_control = num_control
		self.num_state = x_0.shape[0]

	def set_cost(self, c):
		self.c = c

	def set_final_cost(self, c_f):
		self.c_f = c_f

	def set_f(self, f):
		self.f = f

point_dynamics = PointMassTransitionFunction(0.1)
quadratic_cost = QuadraticCostFunction(0 * np.identity(4), 1*np.identity(2))
quadratic_final_cost = QuadraticCostFinalFunction(100*np.identity(4))
t = TrajectoryOptimization(np.array([2.0, 2.0, 0.0, 0.0]), 2, 100, c=quadratic_cost, c_f=quadratic_final_cost, f=point_dynamics)
t.add_state_constraint(CircleConstraintFunction(np.ones(2), 0.5))
t.set_conctrol_constraint(0.2 * np.ones(2))
states, control = t.optimize()
#t.add_state_constraint(CircleConstraintFunction(np.ones(2), 0.5))

for i in range(100):
	print "checking control", control[:, i]

fig,ax = plt.subplots()
ax.scatter(states[0,:],states[1,:])
circle1 = plt.Circle((1, 1), 0.5, color='r')
ax.add_artist(circle1)
plt.axis('equal')
plt.axis([-1, 3, -1, 3])
plt.show()