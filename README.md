# TrajectoryOptimization
trajectory optimization using SNOPT. With constrained expressed with Neural Net.

SNOPT python wrapper can be downloaded from https://github.com/snopt/snopt-python.

The neural network constraint is implemented using pytorch.

optimal_control.py contains the class SQP, which calls SNOPT to solve a simple trajectory optimization with state constraints.

It is tested with a simple 2D point mass dynamic example, with circle constraint.