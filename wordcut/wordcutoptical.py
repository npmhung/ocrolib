import logging
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt

from ocrolib.preprocess.morphology import enhance
# from ocrolib.table_cell_cut.kernel_density_estimate import kernel_density_estimate
# from ocrolib.table_cell_cut.morphology import enhance
from ocrolib.table_cell_cut.kernel_density_estimate import kernel_density_estimate
from ocrolib.wordcut.extras import findContour
from ocrolib.wordcut.wordutils import trimLine

logging.basicConfig(level=logging.INFO)
sys.path.append('/home/warrior/Project/Photoreader')


def show_img(winname, img):
	cv2.imshow(winname, img)
	cv2.waitKey(0)

def show_img(winname, img):
	cv2.imshow(winname, img)
	cv2.waitKey(0)

def reduction(prb, pivot_x, minima, maxima, nearest_size, estimate_wid):
	reduced_pivot = pivot_x
	len_pivot = len(pivot_x)
	'''thresh for two maxima nearest'''
	thresh_gap_width = 10
#print('pp ', pivot_x)
	if len(pivot_x) == 0: return reduced_pivot
	if len(pivot_x) == 1: return pivot_x
	for i in range(1, len_pivot):
		'''select Min pivot '''
		print('pivot x', pivot_x[i])
		distance = abs(pivot_x[i] - pivot_x[i - 1])
		if (distance <= nearest_size and distance <= estimate_wid):
			arg_min = np.argwhere(pivot_x[i])
			max_left = maxima[arg_min]
			max_right = maxima[arg_min+1]
			if abs(max_right- max_left)< thresh_gap_width or abs(max_left - maxima[arg_min-1]):
				print('yes')
				if prb[pivot_x[i]]<= prb[pivot_x[i-1]]:
					reduced_pivot[i-1]=0
				else:
					reduced_pivot[i] = 0

	list_arg = np.nonzero(reduced_pivot)
	reduced_pivot = reduced_pivot[list_arg]
#print('rdpv ', reduced_pivot)
	return reduced_pivot

# plt.show()

'''remove local mimum X, which has histogram projecting value H(X)- H(Y)< thresh, Y is local contigious maximums'''


def check_smooth_hist(list_min, list_max, hist, height):
	thresh = 0.05* height
	cf_interval = 11
#print('list min max len ', len(list_min), ' ', len(list_max))
#print('thresh smooth ', thresh)
	new_list_min = np.array([], dtype=np.int32)
	for arg_min, min in enumerate(list_min):
		try:
			max_left = list_max[arg_min]
			max_right = list_max[arg_min+1]
			print('max left ', max_left, 'max right ', max_right, 'min ', hist[min], ' ',
			      hist[max_left] - hist[min],' ', hist[max_right] - hist[min])
			print('cf ', abs(max_right - max_left))
			if ((hist[max_left] - hist[min] > thresh) or (hist[max_right] - hist[min] > thresh))\
					and (abs(max_right- max_left)> cf_interval) :
				if(abs(max_left- list_max[arg_min+2])> 20):
					new_list_min = np.append(new_list_min, min)
		except:
			new_list_min = np.append(new_list_min, min)
	return new_list_min

def trimming(hist, hei, wid):
	lower, upper = 0, hei - 1
	if(hist[lower] > wid/2):
		for i in reversed(range(hei - 5)):
			if (hist[i-1] > hist[i]+3):
				lower = i
				break
	else:
		for i in reversed(range(hei - 5)):
			if (hist[i] > 3):
				lower = i
				break

	if(hist[upper] > wid/2):
		for i in range(5, hei):
			if (hist[i+1] > hist[i]+4):
				upper = i
				break
	else:
		for i in range(5, hei):
			if (hist[i] >= 5):
				upper = i
				break

	list_arg = [lower, upper]
	return list_arg

def horizontal_pivot(img, table_file, plot):
	copy_img = img.copy()
	dimension = 25
	erosion = enhance(copy_img, table_file, 1, dimension)
	# gray = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
	hei, wid = erosion.shape
#print('shape', hei, wid)
	_, norm_img = cv2.threshold(erosion, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	'''projecting'''
	hist = np.sum(norm_img, axis=0, dtype=np.int32)
	invert_hist = hei - hist
#print('hi', wid)
	logging.info('hist')
	list_bin = np.arange(0, wid)
	calculus_hist = np.repeat(list_bin, invert_hist)

	'''extremas of probability density'''
	is_blank = False
	try:
		mi, ma, s, prb = kernel_density_estimate(calculus_hist, wid, 1)
	except:
		is_blank = True
		return None, erosion, is_blank
	estimate_wid = np.std(ma)
#print('estimate wid ', estimate_wid)
	list_arg_pivot_min = mi
	logging.info('list arg pivot min')
	logging.info(len(list_arg_pivot_min))

	''' smooth pivot'''
	logging.info('check smoothed hist')
	smoothed_bin = check_smooth_hist(list_arg_pivot_min, ma, invert_hist, hei)
	logging.info(len(smoothed_bin))
	logging.info(smoothed_bin)

	nearest_neighbor = 13
	reduce_arg_min = reduction(invert_hist, smoothed_bin, mi, ma, nearest_neighbor, estimate_wid)
	# reduce_arg_min = smoothed_bin
	logging.info(len(reduce_arg_min))
	logging.info('list reduced arg min')
	logging.info(reduce_arg_min)

	'''plot'''
	if (plot):
		fig = plt.figure()
		ax1 = plt.subplot2grid((2, 3), (0, 0))
		ax1.set_xlim([0, wid])
		plt.hist(calculus_hist, bins=50, normed=1)
		plt.legend()
		plt.plot(s, prb)

		ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
		ax2.set_xlim([0, wid])
		plt.plot(s, prb, 'b',
		         s[ma], prb[ma], 'go',
		         s[mi], prb[mi], 'ro')

		for i in reduce_arg_min:
			cv2.line(norm_img, (i, hei), (i, 0), (255, 0, 0), 1)
		ax3 = plt.subplot2grid((2, 3), (1, 0))
		ax3.set_xlim([0, wid])
		plt.imshow(erosion)

		for i in reduce_arg_min:
			cv2.line(copy_img, (i, hei), (i, 0), (255, 0, 0), 1)
		ax5 = plt.subplot2grid((2, 3), (1, 1),colspan=2)
		ax5.set_xlim([0, wid])
		plt.imshow(copy_img)

		figManager = plt.get_current_fig_manager()
		figManager.window.showMaximized()
		plt.tight_layout()
		plt.show()
		# fig.savefig('{}.PNG'.format(table_file[:-4]))
	return reduce_arg_min, erosion, is_blank


def vertical_pivot(img, table_file, plot):
	dimension = 15
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	hei, wid, _ = img.shape
#print('shape', hei, wid)
	_, norm_img = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	'''projecting'''
	hist = np.sum(norm_img, axis=1, dtype=np.int32)
	invert_hist = wid - hist
	# logging.info('hei', hei)
	list_bin = np.arange(0, hei)
	# calculus_hist = np.repeat(list_bin, invert_hist)

	'''extremas of probability density'''
	# mi, ma, s, prb = kernel_density_estimate(calculus_hist, hei, 1)
	# thresh_interval = 5
	# list_arg_pivot_min = mi[mi > thresh_interval]
	# nearest_neighbor = 25
	list_arg = trimming(invert_hist, hei, wid)
	crop_img = img[list_arg[1]:list_arg[0],:]
	# reduce_arg_min = mi[arg_pivot]
#print(len(list_arg))
	logging.info('list reduced arg min')
	logging.info(list_arg)

	'''plot'''
	if (plot):
		'''histogram'''
		fig = plt.figure()
		ax1 = plt.subplot(211)
		# ax1.set_xlim([0, wid])
		ax1.set_ylim([hei, 0])
		# plt.hist(calculus_hist, bins=50, normed=1, orientation="horizontal")
		plt.legend()
		plt.plot(invert_hist, list_bin)

		'''image'''
		for i in list_arg:
			cv2.line(img, (wid, i), (0, i), (255, 0, 0), 1)
		ax5 = plt.subplot(212)
		ax5.set_xlim([0, wid])
		ax5.set_ylim([hei, 0])
		plt.imshow(img)

		figManager = plt.get_current_fig_manager()
		figManager.window.showMaximized()
		plt.tight_layout()
		plt.show()
		fig.savefig('{}v.png'.format(table_file[:-4]))
	return crop_img

def trim_img(img, erosed_img):
#print('trim cell')
	# gray = cv2.cvtColor(erosed_img, cv2.COLOR_RGB2GRAY)
	hei, wid, _ = img.shape
#print('shape', hei, wid)
	_, norm_img = cv2.threshold(erosed_img, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	'''projecting'''
	hist = np.sum(norm_img, axis=0, dtype=np.int32)
	invert_hist = hei - hist
	# print('invert ', invert_hist)
	left, right = 0, hei-1
	for i in range(0,wid):
		if invert_hist[i]> 3:
			left = i
			break
	for i in reversed(range(0,wid)):
		if invert_hist[i]> 3:
			right = i
			break
#print('leftright ', left, right)
	return img[:, left: right]

def trimWord(img):
	list_coor = findContour(img)
	len, _ = list_coor.shape
	sort_left_cts = list_coor[list_coor[:, 0].argsort()].astype(np.int32)
	left = 0
	if len > 0:
		left_ct = sort_left_cts[0]
	# bottom_ct = sort_left_cts[-1]
		left = left_ct[0]
		print('left, right ', left_ct)

	right = img.shape[1]
	right_cts = list_coor[:, 0] + list_coor[:, 2]
	# sort_right_cts = list_coor[right_cts.argsort()].astype(np.int32)
	right_cts = np.sort(right_cts).astype(np.int32)
	if len > 0:
		right_ct = right_cts[-1]
		# right_c = right_cts[0]
		print('right ', right_ct)
		right = right_cts[0]
	trimmed_img = img[:, left: right]
	# plt.imshow(trimmed_img)
	# plt.show()
	return trimmed_img

def wordCutOptical(img, filename, line_file, cell_index, cut_dir):
	logging.info('cell %02i', cell_index)
	if img is None:
		img = cv2.imread(filename)
	hei, wid, _ = img.shape
	# plt.imshow(img_resize)
	# plt.show()
	crop_img = trimLine(img)
	hei_crop, wid_crop, _ = crop_img.shape
	plt.imshow(crop_img)
	plt.title('crop')
	plt.show()
#print('crop size ', crop_img.shape)
	'''normalize all to a size'''
	try:
		img_resize = cv2.resize(crop_img, (int(wid_crop * 40 / hei_crop), 40))
		print('flag1')
	except:
		print('flag2')
		img_resize = cv2.resize(img, (int(wid_crop * 40 / hei_crop), 40))
	plot = 0
	horizontal_arg, erose_img, is_blank = horizontal_pivot(img_resize, filename, plot)
	if is_blank:
		cv2.imwrite('{}{}.blank.png'.format(cut_dir, line_file[:-4]), img)
		return

	resized_hei, resized_wid,_  = img_resize.shape
	full_horizontal_arg = np.hstack((0,horizontal_arg,resized_wid-1))
	for i, pivot in enumerate(full_horizontal_arg[:-1]):
		print('i ', i)
		cut_img = img_resize[:, pivot:full_horizontal_arg[i + 1]]
		# erose_word_img = erose_img[:, pivot: full_horizontal_arg[i+1]]
		try:
			check_blank = check_blank_img(img_resize)
		except:
			print('abc', pivot, ' ', full_horizontal_arg[i+1])
			check_blank = True
		'''check if cut word is wrong by whether it has just blank space'''
		if(not check_blank):
			print('not blank')
			trim_image = cut_img
			'''trim left or right img if there blank space in each word cut'''
			if(full_horizontal_arg[i+1]-pivot> 35):
				trim_image = trimWord(cut_img)
			cv2.imwrite('{}{}.png'.format(cut_dir, i), trim_image)

'''cut cell after have vertical and horizontal pivots'''
def cut_cells(img, list_pivot_x, list_pivot_y, wid, hei):
	full_list_x = np.concatenate(([0], list_pivot_x, [wid - 1]))
	full_list_y = np.concatenate(([0], list_pivot_y, [hei - 1]))
#print('full', full_list_x.shape, " ", full_list_y.shape)
	list_cells_coordinates = []
	for i, pivot_x in enumerate(full_list_x[:-1]):
		print('i', i)
		for j, pivot_y in enumerate(full_list_y[:-1]):
			print('j', j)
			# cell_img = img[pivot_y: full_list_y[j+1],pivot_x: full_list_x[i+1]]
			top_left = (pivot_x, pivot_y)
			top_right = (full_list_x[i + 1], pivot_y)
			bottom_left = (pivot_x, full_list_y[j + 1])
			bottom_right = (full_list_x[i + 1], full_list_y[j + 1])
			print('xy', top_left, top_right, bottom_left, bottom_right)
			cell_coordinate = (top_left[1], top_left[0], bottom_right[1], bottom_right[0])
			list_cells_coordinates.append(cell_coordinate)

#print('len ', len(list_cells_coordinates))
	return list_cells_coordinates

def check_blank_img(img):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	invert_img = cv2.bitwise_not(gray)
	norm_img = cv2.normalize(invert_img,None, 0,1, cv2.NORM_MINMAX)
	sum = np.sum(norm_img)
#print('sum ', sum)
	if sum< 15:
		print('True')
		return True
	else:
		print('False')
		return False

def words_cut_all(dir):
	'''read image and perform cut cells'''
#print('dir', dir)
	# cell_file = '25.png'
	# print('table file', cell_file)
	count = 0
	# for line_file in os.listdir(dir):
	# for line_file in ['01.0001.06.png']:
	# for line_file in ['region_7.png']:
	for line_file in ['01.0001.08.png']:
		count+=1
		file = '{}{}'.format(dir,line_file)
		# file = '{}{}'.format(dir, table_file)
		print('line_fine ', line_file)
		wordCutOptical(None, file, line_file, None, None)
#print('count ', count)

# dir = '/media/warrior/MULTIMEDIA/Newworkspace/Nissay/output/wordcut/one_line/'
# words_cut_all(dir)
