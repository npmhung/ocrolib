import logging
import os
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from ocrolib.preprocess.morphology import enhance
from ocrolib.table_cell_cut.kernel_density_estimate import kernel_density_estimate

logging.basicConfig(level = logging.INFO)
sys.path.append('F:/Hellios-workspace/Photoreader')

def show_img(winname, img):
	cv2.imshow(winname, img)
	cv2.waitKey(0)

def reduction(pivot_x, nearest_size):
	reduced_pivot = []
	if len(pivot_x) == 0: return reduced_pivot
	if len(pivot_x) == 1: return pivot_x
	Min = pivot_x[0]
	for i in range(1,len(pivot_x)):
		'''select Min pivot '''
		print('pivot x', pivot_x[i])
		if abs(pivot_x[i]-pivot_x[i-1])<= nearest_size:
			print('min', min)
			Min = min(pivot_x[i], Min)
		else:
			reduced_pivot.extend([Min])
			Min = pivot_x[i]
		if (i == len(pivot_x) - 1):
			reduced_pivot.extend([Min])
			print('Min1', Min)
	return reduced_pivot

'''cut cells from table which has tab separates keys and values'''
def kde():
	n_basesample = 1000
	np.random.seed(8765678)
	# xn = np.random.randn(n_basesample)
	xn = np.array([10,10,10,20,40,100,50,30,30,10])
	print('xn', np.amax(xn))
	gkde = stats.gaussian_kde(xn)
	ind = np.linspace(-7, 7, 10)
	kdepdf = gkde.evaluate(ind)
	print('norm ', stats.norm.pdf(ind).shape)
	plt.figure()
	# plot histgram of sample
	plt.hist(xn, bins=10, normed=1)
	# plot data generating density
	plt.plot(ind, stats.norm.pdf(ind), color="r", label='DGP normal')
	# plot estimated density
	plt.plot(ind, kdepdf, label='kde', color="g")
	plt.title('Kernel Density Estimation')
	plt.legend()
	# plt.show()

'''remove local mimum X, which has histogram projecting value H(X)- H(Y)< thresh, Y is local contigious maximums'''
def check_smooth_hist(list_min, list_max, hist):
	new_list_min = np.array([], dtype= np.int32)
	thresh = 20
	for min in list_min:
		for j, max in enumerate(list_max):
			if max > min:
				if(hist[max]- hist[min] > thresh)&(hist[list_max[j-1]]- hist[min] > thresh):
					new_list_min= np.append(new_list_min, min)
	return new_list_min


def horizontal_pivot(img, table_file, plot):
	# img = cv2.imread(table_file)
	# img1 = np.asarray(img.copy())
	erosion = enhance(img, table_file, 0)
	# gray = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
	hei, wid = erosion.shape
	print('shape', hei, wid)
	_, norm_img = cv2.threshold(erosion, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	'''projecting'''
	hist = np.sum(norm_img, axis=0, dtype=np.int32)
	invert_hist = hei - hist
	print('hi', wid)
	logging.info('hist')
	logging.info(hist)
	list_bin = np.arange(0, wid)
	num_ones = np.count_nonzero(norm_img)
	calculus_hist = np.repeat(list_bin, invert_hist)

	'''extremas of probability density'''
	mi, ma, s, prb = kernel_density_estimate(calculus_hist, wid, 20)
	thresh_interval_right = 160
	thresh_interval_left = 120
	list_arg_pivot_min = mi[(mi< wid- thresh_interval_right) & (mi >  thresh_interval_left)]
	logging.info('list arg pivot min')
	logging.info(list_arg_pivot_min.shape)
	nearest_neighbor = 120

	'''threshold for get low value pivot'''
	max_invert_hist = max(invert_hist[ma])
	logging.info('max_invert_hist')
	logging.info(max_invert_hist)
	
	'''filter low hist value '''
	list_arg_pivot_min = list_arg_pivot_min[invert_hist[list_arg_pivot_min]< max_invert_hist*2/3]
	logging.info('list arg pivot min')
	logging.info(list_arg_pivot_min)
	reduce_arg_min = reduction(list_arg_pivot_min, nearest_neighbor)
	print(len(reduce_arg_min))
	logging.info('list reduced arg min')
	logging.info(reduce_arg_min)

	''' smooth pivot'''
	smoothed_bin = check_smooth_hist(reduce_arg_min, ma, invert_hist)
	logging.info('check smoothed hist')
	logging.info(smoothed_bin)

	'''plot'''
	if (plot):
		fig = plt.figure()
		ax1 = plt.subplot(231)
		ax1.set_xlim([0, wid])
		plt.hist(calculus_hist, bins=50, normed=1)
		plt.legend()
		plt.plot(s, prb)

		ax2 = plt.subplot(232)
		ax2.set_xlim([0, wid])
		plt.plot(s, prb, 'b',
				 s[ma], prb[ma], 'go',
				 s[mi], prb[mi], 'ro')

		for i in smoothed_bin:
			cv2.line(norm_img, (i, hei), (i ,0), (255,0,0), 2)
		ax3 = plt.subplot(233)
		ax3.set_xlim([0, wid])
		plt.imshow(erosion)

		ax4 = plt.subplot(234)
		ax4.set_xlim([0, wid])
		plt.plot(list_bin, invert_hist, 'b',
				list_bin[smoothed_bin], hist[smoothed_bin], 'ro')

		for i in smoothed_bin:
			cv2.line(img, (i, hei), (i ,0), (255,0,0), 4)
		ax5 = plt.subplot(235)
		ax5.set_xlim([0, wid])
		plt.imshow(img)

		figManager = plt.get_current_fig_manager()
		figManager.window.showMaximized()
		plt.tight_layout()
		plt.show()
	return smoothed_bin

def vertical_pivot(img, table_file, plot):
	erosion = enhance(img, table_file, 1)
	hei, wid = erosion.shape
	print('shape', hei, wid)
	_, norm_img = cv2.threshold(erosion, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	'''projecting'''
	hist = np.sum(norm_img, axis=1, dtype=np.int32)
	invert_hist = wid - hist
	logging.info('hei', hei)
	list_bin = np.arange(0, hei)
	num_ones = np.count_nonzero(norm_img)
	calculus_hist = np.repeat(list_bin, invert_hist)

	'''extremas of probability density'''
	mi, ma,  s, prb = kernel_density_estimate(calculus_hist, hei, 4)
	thresh_interval = 50
	list_arg_pivot_min = mi[mi > thresh_interval]
	nearest_neighbor = 20
	'''threshold for get low value pivot'''
	max_invert_hist = max(invert_hist[ma])
	print('max_invert_hist', max_invert_hist)
	list_arg_pivot_min = list_arg_pivot_min[invert_hist[list_arg_pivot_min]< max_invert_hist/2]

	reduce_arg_min = reduction(list_arg_pivot_min, nearest_neighbor)
	'''sorting pivot'''
	'''adjust highest pivot'''
	if len(reduce_arg_min) >= 1:
		reduce_arg_min[0]-=10
	print(len(reduce_arg_min))
	logging.info('list reduced arg min')
	logging.info(list_arg_pivot_min)

	'''plot'''
	if(plot):
		'''histogram'''
		fig = plt.figure()
		ax1 = plt.subplot(231)
		# ax1.set_xlim([0, wid])
		ax1.set_ylim([hei, 0])
		plt.hist(calculus_hist, bins=50, normed=1,  orientation="horizontal")
		plt.legend()
		plt.plot(prb, s)

		'''kde'''
		ax2 = plt.subplot(232)
		# ax2.set_xlim([0, wid])
		ax2.set_ylim([hei,0])
		plt.plot(prb,s, 'b',
				 prb[ma], s[ma], 'go',
				 prb[mi], s[mi],'ro')

		'''erosion image'''
		for i in reduce_arg_min:
			cv2.line(norm_img, (wid, i), (0 ,i), (255,0,0), 2)
		ax3 = plt.subplot(233)
		ax3.set_xlim([0, wid])
		ax3.set_ylim([hei,0])
		plt.imshow(erosion)

		''''''
		# ax4 = plt.subplot(234)
		# ax4.set_xlim([0, wid])
		# plt.plot(list_bin, invert_hist, 'b',
		# 		 # list_bin[mi], hist[mi], 'ro',
		# 		list_bin[reduce_arg_min], hist[reduce_arg_min], 'ro')

		'''image'''
		for i in reduce_arg_min:
			cv2.line(img, (wid, i), (0 ,i), (255,0,0), 4)
		ax5 = plt.subplot(235)
		ax5.set_xlim([0, wid])
		ax5.set_ylim([hei, 0])
		plt.imshow(img)

		figManager = plt.get_current_fig_manager()
		figManager.window.showMaximized()
		plt.tight_layout()
		plt.show()
	return reduce_arg_min

def tab_cut_cells(im_table):
	img = np.array(im_table * 255, dtype=np.uint8)
	horizontal_arg = horizontal_pivot(img, None, None)
	vertical_arg = vertical_pivot(img, None, None)
	hei, wid = img.shape
	return cut_cells(img, horizontal_arg, vertical_arg, wid, hei)
	# for i in horizontal_arg:
	# 	cv2.line(img, (i, hei), (i ,0), (255,0,0), 4)
	#
	# for i in vertical_arg:
	# 	cv2.line(img, (wid, i), (0, i), (255,0,0), 4)
	#
	#show_img('',img)
	# cv2.imwrite('{}/Data/output/table/cells/{}/{}'.format(sys.path[-1],format, filename), img)

def cut_cells(img, list_pivot_x, list_pivot_y, wid, hei):
	full_list_x = np.concatenate(([0],list_pivot_x,[wid-1]))
	full_list_y = np.concatenate(([0], list_pivot_y, [hei-1]))
	print('full', full_list_x.shape," ", full_list_y.shape)
	list_cells_coordinates = []
	for i,pivot_x in enumerate(full_list_x[:-1]):
		print('i', i)
		for j,pivot_y in enumerate(full_list_y[:-1]):
			print('j', j)
			#cell_img = img[pivot_y: full_list_y[j+1],pivot_x: full_list_x[i+1]]
			top_left = (pivot_x, pivot_y)
			top_right = (full_list_x[i+1], pivot_y)
			bottom_left = (pivot_x, full_list_y[j+1])
			bottom_right = (full_list_x[i+1], full_list_y[j+1])
			print('xy', top_left, top_right, bottom_left, bottom_right)
			cell_coordinate = (top_left[1], top_left[0], bottom_right[1], bottom_right[0])
			list_cells_coordinates.append(cell_coordinate)

	print('len ', len(list_cells_coordinates))
	return list_cells_coordinates

def cut_cells_all_tables(dir):
	'''read image and perform cut cells'''
	print('dir', dir)
	# for table_file in os.listdir(dir):
	# table_file = '1.PNG'
	# table_file = '2.PNG'
	# table_file = '3.PNG'
	# table_file = '4.PNG'
	# table_file = '5.PNG'
	# table_file = '6.PNG'
	table_file = '7.PNG'
	# table_file = '8.PNG'
	# table_file = '9.PNG'
	print('table file', table_file)
	for table_file in [table_file]:
		table_path = '{}{}'.format(dir,table_file)
		if os.path.isfile(table_path):
			file = '{}/Data/output/{}/cut/{}'.format(sys.path[-1], table_file)
			tab_cut_cells(file,table_file)

