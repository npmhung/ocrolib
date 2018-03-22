import numpy as np
from sklearn.neighbors.kde import KernelDensity
from scipy.signal import argrelextrema

def kernel_density_estimate(histogram, num_samples, bandwidth):
	kernel = 'gaussian'
	# kernel = 'epanechnikov'
	kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(histogram.reshape(-1, 1))
	# kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(a)
	# s = np.linspace(0, 50)
	# v = np.linspace(0,10,10)
	# plt.plot(v, a)
	# plt.show()
	# wid = 831
	s = np.linspace(0, num_samples, num_samples)
	# e = kde.score_samples(s.reshape(-1, 1))
	e = kde.score_samples(s.reshape(-1, 1))
	prb = np.exp(e)

	# plt.figure()
	# print('e', e.shape, e)
	# print('prb', prb.shape, prb)
	'''reduce '''
	# reduced_hist = np.unique(hist)
	# num_bin = len(reduced_hist)

	# print('num_bin', num_bin)


	'''extremas of probability density'''
	mi, ma = argrelextrema(prb, np.less)[0], argrelextrema(prb, np.greater)[0]
	# print("Minima:", mi)
	# print("Maxima:", ma)
	return mi,ma, s, prb