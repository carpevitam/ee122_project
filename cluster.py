import numpy as np


'''
a cluster of points
each point is associated with a value
'''
class Cluster:

	'''
	initialize a cluster at point
	'''
	def __init__(self, point, value):
		self.points = [point]
		self.v_sum = np.asarray(value)
		self.count = 1

	def mean(self):
		return np.divide(self.v_sum, self.count)

	def add(self, point, value):
		self.points.append(point)
		self.v_sum += np.asarray(value)
		self.count += 1