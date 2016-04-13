from math import sqrt
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import random

DIFF_THRESH = 2
ITER_THRESH = 100




class MeanShift:
	
	def __init__(self, image_name, hs, hr):
		self.image = misc.imread(image_name)
		self.result = misc.imread(image_name)
		self.modes = dict()
		self.dim = self.image.shape[2] + 2 # this is the dimension of the feature space, typically 5 with (x, y, r, g, b)
		self.hr = hr # color range
		self.hs = hs # spatial range

	'''
	Calculate the difference between two feature space points.
	This function assumes v1 and v2 are of dtype int
	'''
	def diff(self, v1, v2):
		r = np.linalg.norm(v1-v2)
		return r

	'''
	Calculate the mean around a vector in the feature space
	This function assumes vector dtype is int
	'''
	def mean(self, vector):

		count = 0
		x = vector[0]
		y = vector[1]
		rgb = vector[2:]
		v_sum = np.zeros(self.dim).astype(int)

		# spatial window
		start_x = max(x - self.hs, 0)
		end_x = min(self.image.shape[0] - 1, x + self.hs)
		start_y = max(y - self.hs, 0)
		end_y = min(self.image.shape[1] - 1, y + self.hs)

		for i in range(start_x, end_x):
			for j in range(start_y, end_y):
				_rgb = self.image[i,j]
				diff = self.diff(_rgb, rgb)
				if diff <= self.hr:
					count += 1
					v_sum += np.concatenate([[i,j],_rgb])

		if count == 0:
			return vector
		else:
			return np.divide(v_sum, count)

	'''
	find the mode of the pixel (x, y) in the image
	using the mean shift algorithm
	'''
	def shift(self, x, y):
		iteration = 0
		cur = np.concatenate([[x, y], self.image[x,y]]).astype(int)
		next = self.mean(cur)
		while self.diff(cur, next) > DIFF_THRESH:
			cur = next
			next = self.mean(cur)
			iteration+=1
			if(iteration > ITER_THRESH):
				print 'ITER_THRESH reached when shifting ' + str(x) + ', ' + str(y)
				break
		return next


	'''
	Launch the process by first shifting every pixel to its mode,
	and then group similar modes into clusters.
	'''
	def run(self):
		print 'Begin shifting on ' + str(self.image.shape[0]) + ' * ' + str(self.image.shape[1]) + ' image. '
		print 'Paramter: hs = ' + str(self.hs) + '; hr = ' + str(self.hr) + '; dim = ' + str(self.dim)
		for i in range(self.image.shape[0]):
			print 'shifting row ' + str(i)
			for j in range(self.image.shape[1]):
				mode = self.shift(i, j)
				self.modes[(i, j)] = mode
				self.result[i, j] = mode[2:]

		print 'Begin clustering.'

		clusters = []

		for i in range(self.image.shape[0]):
			print 'clustering row ' + str(i)
			for j in range(self.image.shape[1]):
				self.add_to_clusters(clusters, (i, j))

		print 'number of clusters: ' + str(len(clusters))

		for c in clusters:
			color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
			for point in c:
				self.result[point[0], point[1]] = color



	def belong_to_cluster(self, c, mode):
		x, y = mode[0], mode[1]
		rgb = mode[2:]
		for point in c:
			m = self.modes[point]
			m_x, m_y = m[0], m[1]
			m_rgb = m[2:]
			if self.diff(rgb, m_rgb) > self.hr:
				return False
		return True

	def add_to_clusters(self, clusters, point):
		for c in clusters:
			if self.belong_to_cluster(c, self.modes[point]):
				c.append(point)
				return
		clusters.append([point])
		return


	def save(self):
		misc.imsave('result.png', self.result)

	def show(self):
		fig = plt.figure()
		a=fig.add_subplot(1,2,1)
		imgplot = plt.imshow(self.image)
		a.set_title('Original')
		a=fig.add_subplot(1,2,2)
		imgplot = plt.imshow(self.result)
		a.set_title('Segmented')
		plt.show()



m = MeanShift('test4.jpg', hs = 10, hr = 30)
m.run()
m.save()
m.show()
