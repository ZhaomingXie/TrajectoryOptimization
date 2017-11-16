import numpy        as np
import scipy.sparse as sp

class StateControlFunction(object):
	def __init__(self):
		print "Function Initialized"
	def f(self, x, u):
		return None
	def f_x(self, x, u):
		return None
	def f_u(self, x, u):
		return None

class StateFunction(object):
	def __init__(self):
		print "Function Initialized"
	def f(self, x):
		return None
	def f_x(self, x):
		return None

class LinearTransitionFunction(StateControlFunction):
	def __init__(self, A, B):
		super(StateControlFunction, self).__init__()
		self.A = A
		self.B = B
	def f(self, x, u):
		return self.A.dot(x) + self.B.dot(u)
	def f_x(self, x, u):
		return self.A
	def f_u(self, x, u):
		return self.B

class QuadraticCostFunction(StateControlFunction):
	def __init__(self, Q, R):
		super(StateControlFunction, self).__init__()
		self.Q = Q
		self.R = R
	def f(self, x, u):
		return x.T.dot(self.Q).dot(x) + u.T.dot(self.R).dot(u)
	def f_x(self, x, u):
		return 2 * self.Q.dot(x)
	def f_u(self, x, u):
		return 2 * self.R.dot(u)
	def f_xx(self, x, u):
		return 2 * self.Q
	def f_xu(self, x, u):
		return np.zeros((x.shape[0], u.shape[0]))
	def f_uu(self, x, u):
		return 2 * self.R

class QuadraticCostFinalFunction(StateFunction):
	def __init__(self, Q):
		super(StateFunction, self).__init__()
		self.Q = Q
	def f(self, x):
		return x.T.dot(self.Q).dot(x)
	def f_x(self, x):
		return 2 * self.Q.dot(x)
	def f_xx(self, x):
		return 2 * self.Q

class PointMassTransitionFunction(LinearTransitionFunction):
	def __init__(self, h):
		A = np.identity(4)
		A[0, 2] = h
		A[1, 3] = h
		B = np.zeros((4, 2))
		B[2, 0] = h
		B[3, 1] = h
		super(LinearTransitionFunction, self).__init__()
		self.A = A
		self.B = B

class CircleConstraintFunction(StateFunction):
	def __init__(self, center, r):
		super(StateFunction, self).__init__()
		self.center = center
		self.r = r

	def f(self, x):
		return self.r**2 - (x[0:2] - self.center).T.dot(x[0:2] - self.center)

	def f_x(self, x):
		result = np.zeros(x.shape)
		result[0:2] = -2 * x[0:2] + 2 * self.center
		return result