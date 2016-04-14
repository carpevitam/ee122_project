from math import sqrt
from scipy import misc
from cluster import Cluster
import numpy as np
import matplotlib.pyplot as plt

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
		self.hr_filter = self.hr_filter()

	'''
	Calculate the difference between two feature space points.
	This function assumes v1 and v2 are of dtype int
	'''
	def diff(self, v1, v2):
		r = np.linalg.norm(v1-v2)
		return r



	'''
	return a function that filters out value greater than self.hr
	'''
	def hr_filter(self):
		def f(diff):
			if diff > self.hr: 
				return 0
			else:
				return 1
		return np.vectorize(f)


	'''
	Calculate the mean around a vector in the feature space
	This function assumes vector dtype is int
	'''
	def mean(self, vector):

		# count = 0
		x = vector[0]
		y = vector[1]
		rgb = vector[2:]
		v_sum = np.zeros(self.dim).astype(int)

		# spatial window
		start_x = max(x - self.hs, 0)
		end_x = min(self.image.shape[0] - 1, x + self.hs)
		start_y = max(y - self.hs, 0)
		end_y = min(self.image.shape[1] - 1, y + self.hs)

		window_1 = self.image[start_x: end_x + 1, start_y: end_y + 1]
		window_2 = np.zeros((end_x - start_x + 1, end_y - start_y + 1, self.dim - 2))
		for i in range(window_2.shape[0]):
			for j in range(window_2.shape[1]):
				window_2[i, j] = rgb


		diff_window = np.apply_along_axis(np.linalg.norm, 2, window_1 - window_2)
		diff_window = self.hr_filter(diff_window) # (i, j) = 1 if the point contribute to the mean

		count = 0

		for i in range(window_2.shape[0]):
			for j in range(window_2.shape[1]):
				if diff_window[i,j] == 1:
					count += 1
					v_sum += np.concatenate([[start_x + i,start_y + j], window_1[i, j]])

		if count == 0:
			return vector

		return np.divide(v_sum, count)



		# previous code
		# for i in range(start_x, end_x):
		# 	for j in range(start_y, end_y):
		# 		_rgb = self.image[i,j]
		# 		diff = self.diff(_rgb, rgb)
		# 		if diff <= self.hr:
		# 			count += 1
		# 			v_sum += np.concatenate([[i,j],_rgb])

		# if count == 0:
		# 	return vector
		# else:
		# 	return np.divide(v_sum, count)

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

		print 'Begin clustering.'

		clusters = [] # a list of Cluster object

		for i in range(self.image.shape[0]):
			print 'clustering row ' + str(i)
			for j in range(self.image.shape[1]):
				self.add_to_clusters(clusters, (i, j))

		print 'number of clusters: ' + str(len(clusters))

		for c in clusters:
			color = [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]
			for point in c.points:
				self.result[point[0], point[1]] = color


	# exact clustering - slow
	#
	# def belong_to_cluster(self, c, mode):
	# 	x, y = mode[0], mode[1]
	# 	rgb = mode[2:]
	# 	for point in c:
	# 		m = self.modes[point]
	# 		m_x, m_y = m[0], m[1]
	# 		m_rgb = m[2:]
	# 		if self.diff(rgb, m_rgb) > self.hr:
	# 			return False
	# 	return True

	# def add_to_clusters(self, clusters, point):
	# 	for c in clusters:
	# 		if self.belong_to_cluster(c, self.modes[point]):
	# 			c.append(point)
	# 			return
	# 	clusters.append([point])
	# 	return


	# relaxed clustering - fast but maybe inaccurate
	
	def belong_to_cluster(self, c, point, mode):
		rgb = mode[2:]
		c_mean = c.mean()
		c_rgb = c_mean[2:]
		if self.diff(c_rgb, rgb) > self.hr:
			return False
		return True

	def add_to_clusters(self, clusters, point):
		mode = self.modes[point]
		for c in clusters:
			if self.belong_to_cluster(c, point, mode):
				c.add(point, mode)
				return
		clusters.append(Cluster(point, mode))
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

m = MeanShift('test4.jpg', hs = 8, hr = 30)
m.run()
m.save()
m.show()
