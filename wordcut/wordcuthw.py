import logging
import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from matplotlib import pyplot as plt

from ocrolib.preprocess.morphology import enhance
from ocrolib.preprocess.preprocess import denoise
from ocrolib.preprocess.rotate import rotate
from ocrolib.table_cell_cut.kernel_density_estimate import kernel_density_estimate
from ocrolib.wordcut.review_pivots import reviewPivots, find_cpn
from ocrolib.wordcut.wordcutCntcpn import cutCpn
from ocrolib.wordcut.extras import findContour
from ocrolib.wordcut.wordutils import trimLine, checkDateField, checkNoise
from ocrolib.line_seg_util import normalize_cell_img
from scipy import ndimage


def show_img(winname, img):
	# cv2.imshow(winname, img)
	# cv2.waitKey(0)
	plt.imshow(img)
	plt.title(winname)
	plt.show()


def reduction(img, prb, pivot_x, minima, maxima, nearest_size, estimate_wid, config):
	reduced_pivot = pivot_x
	len_pivot = len(pivot_x)
	'''thresh for two maxima nearest'''
	thresh_gap_width = config.getint('reduction','threshGapWidth')
	# recheck_pivot = config.getint('reduction','recheckpivot')
	#print('pp ', pivot_x)
	if len(pivot_x) == 0: return reduced_pivot
	if len(pivot_x) == 1: return pivot_x
	for i in range(1, len_pivot):
		'''select Min pivot '''
		distance = abs(pivot_x[i] - pivot_x[i - 1])
		if (distance <= nearest_size and distance <= estimate_wid):
			arg_min = np.argwhere(pivot_x[i])
			max_left = maxima[arg_min]
			max_right = maxima[arg_min+1]
			if abs(max_right- max_left)< thresh_gap_width:
				# if recheck_pivot:
				# 	if prb[pivot_x[i]]<= prb[pivot_x[i-1]]:
				# 		if not reviewPivots(img, reduced_pivot[i-1]):
				# 			reduced_pivot[i-1]=0
				# 	else:
				# 		if not reviewPivots(img, reduced_pivot[i]):
				# 			reduced_pivot[i] = 0
				# else:
				if prb[pivot_x[i]]<= prb[pivot_x[i-1]]:
					reduced_pivot[i-1]=0
				else:
					reduced_pivot[i] = 0

	list_arg = np.nonzero(reduced_pivot)
	reduced_pivot = reduced_pivot[list_arg]
	#print('rdpv ', reduced_pivot)
	return reduced_pivot


'''remove local mimum X, which has histogram projecting value H(X)- H(Y)< thresh, Y is local contigious maximums'''

def check_smooth_hist(img, list_min, list_max, hist, height, config):
	thresh = config.getfloat('checksmoothhist', 'threshCoef')
	cf_interval = config.getint('checksmoothhist', 'cf_interval')
	recheck_pivot = config.getint('reduction', 'recheckpivot')
	list_cpn = []
	if recheck_pivot:
		list_cpn = find_cpn(img, config)
	#print('list min max len ', len(list_min), ' ', len(list_max))
	#print('thresh smooth ', thresh)
	new_list_min = np.array([], dtype=np.int32)
	for arg_min, min in enumerate(list_min):
		try:
			max_left = list_max[arg_min]
			max_right = list_max[arg_min+1]
			# #print('max left ', max_left, 'max right ', max_right, 'min ', hist[min], ' ',
			#       hist[max_left] - hist[min],' ', hist[max_right] - hist[min])
			# #print('cf ', abs(max_right - max_left))
			'''max left or max right subtraction to min or max left to max right distance'''
			if ((hist[max_left] - hist[min] > thresh) and (hist[max_right] - hist[min] > thresh))\
					and (abs(max_right- max_left)> cf_interval)\
				and (abs(max_left - min) > 5 or abs(max_right - min)> 5):
				'''two maximum with a maximum between'''
				if(abs(max_left- list_max[arg_min+2])> cf_interval):
					new_list_min = np.append(new_list_min, min)

			else:
				if recheck_pivot:
					isCut2Text = reviewPivots(img, list_cpn, min, config)
					if not isCut2Text:
						new_list_min = np.append(new_list_min, min)
		except:
			new_list_min = np.append(new_list_min, min)
	return new_list_min

def checkCrossing(img, list_pivots):
	# print('len', len(list_pivots))
	invert = np.array(1- img, dtype= np.int8)
	uncrossed_pivots = list_pivots
	for index, pivot in enumerate(list_pivots):
		sub_img = invert[:, pivot]
		# print('sub_img', sub_img)
		diff = np.diff(sub_img)
		# print('diff', diff)
		num_cross = (diff == 1).sum()
		# print('numcross', num_cross)
		if num_cross > 1:
			# print('pv', pivot)
			list_pivots[index] = 0
	list_args = np.nonzero(list_pivots)
	uncrossed_pedvots = uncrossed_pivots[list_args]
	# print(len(uncrossed_pivots),uncrossed_pivots)
	return uncrosconfig.readsed_pivots

def trimming(hist, hei, wid, config):
	thresh_diff = config.getint('trimming','threshDif')
	start_traverse = config.getint('trimming', 'startTraverse')
	lower, upper = 0, hei - 1
	if(hist[lower] > wid/2):
		for i in reversed(range(hei - start_traverse)):
			if (hist[i-1] > hist[i]+ thresh_diff):
				lower = i
				break
	else:
		for i in reversed(range(hei - start_traverse)):
			if (hist[i] > thresh_diff):
				lower = i
				break
#b_img.pkl
	if(hist[upper] > wid/2):
		for i in range(start_traverse, hei):
			if (hist[i+1] > hist[i]+4):
				upper = i
				break
	else:
		for i in range(start_traverse, hei):
			if (hist[i] >= 5):
				upper = i
				break

	list_arg = [lower, upper]
	return list_arg

def horizontal_pivot(img, table_file, plot, config):
	copy_img = img.copy()
	dimension = config.getint('horizontalpivot', 'eroseDimension')
	bandwidth = config.getfloat('horizontalpivot', 'bandwidth')
	erosion = enhance(copy_img, table_file, 1, dimension,None, None, config)
	# gray = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
	hei, wid = erosion.shape
	#print('shape', hei, wid)
	_, norm_img = cv2.threshold(erosion, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	'''projecting'''
	hist = np.sum(norm_img, axis=0, dtype=np.int32)
	invert_hist = hei - hist
	# logging.info('hist')
	list_bin = np.arange(0, wid)
	calculus_hist = np.repeat(list_bin, invert_hist)

	'''extremas of probability density'''
	is_blank = False
	try:
		mi, ma, s, prb = kernel_density_estimate(calculus_hist, wid, bandwidth)
	except:
		is_blank = True
		return None, erosion, is_blank
	#print('im her++++++++++++++++++++++++++++++++++++++++')
	estimate_wid = np.std(ma)
	#print('estimate wid ', estimate_wid)
	list_arg_pivot_min = mi
	# logging.info('list arg pivot min')
	# logging.info(len(list_arg_pivot_min))

	# check_cross = config.getint('horizontalpivot', 'checkcross')
	# if check_cross:config.read
	# 	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# 	_, thesh = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# 	list_arg_pivot_min = checkCrossing(thesh, mi)

	''' smooth pivot'''
	# logging.info('check smoothed hist')
	smoothed_bin = check_smooth_hist(copy_img, list_arg_pivot_min, ma, invert_hist, hei, config)
	# logging.info(len(smoothed_bin))
	# logging.info(smoothed_bin)

	nearest_neighbor = config.getint('horizontalpivot', 'nearestNeighbor')
	reduce_arg_min = reduction(copy_img, invert_hist, smoothed_bin, mi, ma, nearest_neighbor, estimate_wid, config)
	# logging.info(reduce_arg_min)
	check_cross = False #config.getint('horizontalpivot', 'checkcross')
	if check_cross:
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		_, thesh = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY| cv2.THRESH_OTSU)
		final_pivot = checkCrossing(thesh, reduce_arg_min)
	else:
		final_pivot = reduce_arg_min
	# logging.info(len(reduce_arg_min))
	# logging.info('list reduced arg min')
	# logging.info(final_pivot)

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
		# prb_unnorm = prb*hei
		plt.plot(s, prb, 'b',
		         s[ma], prb[ma], 'go',
		         s[mi], prb[mi], 'ro')

		# for i in final_pivot:
		# 	cv2.line(norm_img, (i, hei), (i, 0), (255, 0, 0), 1)
		ax3 = plt.subplot2grid((2, 3), (1, 0))
		ax3.set_xlim([0, wid])
		plt.imshow(erosion)

		# ax4 = plt.subplot2grid((2, 3), (1, 0))
		# ax4.set_xlim([0, wid])
		# ax4.set_title('smoothed bin')
		# plt.plot(list_bin, invert_hist, 'b',
		#          list_bin[smoothed_bin], hist[smoothed_bin], 'ro')

		for i in final_pivot:
			cv2.line(copy_img, (i, hei), (i, 0), (255, 0, 0), 2)
		ax5 = plt.subplot2grid((2, 3), (1, 1),colspan=2)
		ax5.set_xlim([0, wid])
		plt.imshow(copy_img)
		figManager = plt.get_current_fig_manager()
		figManager.resize(*figManager.window.maxsize())
		plt.tight_layout()
		plt.show()
		# fig.savefig('{}.PNG'.format(table_file[:-4]))
	return final_pivot, erosion, is_blank


def vertical_pivot(img, table_file, plot, config):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	hei, wid, _ = img.shape
	#lib_img.pklprint('shape', hei, wid)
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

def trimWord(img, config):
	'''trim left and right word'''
	list_coor = findContour(img)
	len, _ = list_coor.shape
	sort_left_cts = list_coor[list_coor[:, 0].argsort()].astype(np.int32)
	left = 0
	if len > 0:
		left_ct = sort_left_cts[0]
		cd_left = sort_left_cts[-1]
		left = left_ct[0]
		#print('left, right ', left_ct,  cd_left)

	right = img.shape[1]
	right_cts = list_coor[:, 0] + list_coor[:, 2]
	right_cts = np.sort(right_cts).astype(np.int32)
	if len > 0:
		right_ct = right_cts[-1]
		#print('right ', right_ct)
		right = right_cts[0]
	trimmed_img = img[:, left: right]
	# plt.imshow(trimmed_img)
	# plt.show()
	return trimmed_img

def trim_img(img, erosed_img, config):
	#print('trim cell')
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	hei, wid, _ = img.shape
	draw_img = img.copy()
	hei_erose, wid_erose = erosed_img.shape
	#print('erose ', wid_erose)
	#print('shape', hei, wid)
	label_im, nb_labels = ndimage.label(gray < 120)
	# show_img('label', label_im)
	labels = np.unique(label_im)
	label_im = np.searchsorted(labels, label_im)
	list_cpn = ndimage.find_objects(label_im)
	# print('listcpn', list_cpn)
	sorted_cpn = sorted(list_cpn, key=lambda x: ((x[1].stop - x[1].start) * (x[0].stop- x[0].start)))
	# print('sorted cpn', sorted_cpn)
	# for i, slice in enumerate(sorted_cpn):
	# 	# slice_x, slice_y = ndimage.find_objects(label_im==i)
	# 	slice_x, slice_y = slice
	# 	tmp_img = img[slice_x,slice_y]
	# 	show_img('tmp ', tmp_img)
	main_cpn = sorted_cpn[-1]
	# print('main',  main_cpn)
	# print('lencpn', nb_labels)
	# rm_area = config.getint('connectedcpn', 'remove_area')
	slice_x, slice_y = main_cpn
	left = min(slice_x.start -3, 0)
	right = max(slice_x.stop +3, wid_erose -1)
	# print('leftright ', left, right)
	cv2.line(draw_img,(left,0), (left,hei-1),(255,0,0),1)
	cv2.line(draw_img, (right, 0), (right, hei - 1), (0, 255, 0), 1)
	# plt.imshow(img[:, left: right])
	# plt.show()
	return img[:, left: right], left, right, draw_img

def word_cut(img, filename, line_file, cell_index, cut_dir, config, plot):
	logging.info('cell %02i', cell_index)
	print(filename)
	if img is None:
		img = cv2.imread(filename)
	hei, wid, _ = img.shape
	#img = denoise(img)
	# show_img('erose', img)
	rotate_img = img
	#is_rotate = config.getint('rotation', 'rotate')
	#print('rotate', is_rotate)
	#if is_rotate:
	#	rotate_img = rotate(img, config)
		# show_img('rotate', rotate_img)
	crop_img =img# trimLine(rotate_img)
	# show_img('trimLine', crop_img)
	hei_crop, wid_crop, _ = crop_img.shape
	resize_coef = config.getint('wordCut', 'resizeCoef')
	thresh_trim = config.getint('wordCut','threshTrim')
	#isDateField = config.getint('datefield', 'isdate')
	#findCpn = config.getint('connectedcpn','findcpn')
	isNumber = config.getint('number', 'isnumbercut')
	#print('int', int(wid_crop * resize_coef / hei_crop))
	#print('crop size ', crop_img.shape)
	'''normalize all to a size'''
	# try:
	# 	img_resize = cv2.resize(crop_img, (int(wid_crop * 40 / hei_crop), 40))
	# 	#print('flag1')
	# except:
	# 	#print('flag2')
	# 	img_resize = cv2.resize(img, (int(wid_crop * 40 / hei_crop), 40))

	try:
		img_resize = img #cv2.resize(crop_img, (wid_crop, resize_coef))
		#print('flag1')
	except:
		#print('flag2')
		img_resize =img# cv2.resize(img, (wid, resize_coef))
	# show_img('re', img_resize)
	# plot = 0
	horizontal_arg, erose_img, is_blank = horizontal_pivot(img_resize, filename, plot, config)

	if is_blank:
		#cv2.imwrite('{}{}.{}.blank.png'.format(cut_dir, line_file[:-4], cell_index), img)
		return


	resized_hei, resized_wid,_  = img_resize.shape
	isDateField = False
	if isDateField:
		pivot_d = int(config.getfloat('datefield', 'd') * wid_crop)
		pivot_m = int(config.getfloat('datefield', 'm') * wid_crop)

	full_horizontal_arg = np.hstack((0,horizontal_arg,resized_wid-1))
	# list_cut_img = []
	index_word = 0
	for i, pivot in enumerate(full_horizontal_arg[:-1]):
		#print('i ', i)
		cut_img = crop_img[:, pivot:full_horizontal_arg[i + 1]]
		erose_word_img = erose_img[:, pivot: full_horizontal_arg[i+1]]
		try:
			check_blank = check_blank_img(cut_img,config)
		except:
			check_blank = True
		# nulib_img.pklm_ct = finding_contour(cut_img, erose_word_img)
		'''check if cut word is wrong by whether it has just blank space'''
		if(not check_blank):
			# #print('not blank')
			trim_image = cut_img
			left = 0
			'''trim left or right img if there blank space in each word cut'''
			if(full_horizontal_arg[i+1]-pivot> thresh_trim):
				trim_image, left, right, draw_img = trim_img(cut_img, erose_word_img, config)
				# plt.imshow(draw_img)
				# plt.show()
				# trim_image = trimWord(cut_img)
			# print('left', left)
			hei_trim, wid_trim,_ = trim_image.shape
			isNoise = checkNoise(hei_trim, wid_trim)
			if isNoise:
				continue
			trim_image = cv2.cvtColor(trim_image, cv2.COLOR_BGR2GRAY)
			trim_image = normalize_cell_img(trim_image)
			if isDateField:
				date_index = checkDateField(crop_img, pivot+ left +3, pivot_d, pivot_m)
				cv2.imwrite('{}{}.{}.png'.format(cut_dir, date_index, index_word), trim_image)
				index_word+=1
			else:
				# list_cpn = cutCpn(trim_image)
				# h, w = trim_image.shape
				# for j, slice_y in enumerate(list_cpn):
				# 	starty = max(slice_y[0] - 2, 0)
				# 	print('sss ', slice_y)
				# 	stopy = min(slice_y[1] + 2, w - 1)
				# 	cpn_img = trim_image[:, starty: stopy]
				# 	# show_img('', cpn_img)
				# 	cv2.imwrite('{}{}.{}.png'.format(cut_dir, cell_index, i+j), cpn_img)
				# 	i+=1
				# if findCpn:
				# 	list_cpn = find_cpn(trim_image, config)
				# 	# print(list_cpn)
				# 	# print('len ', len(list_cpn))
				# 	if len(list_cpn)> 1:
				# 		for j,cpn in enumerate(list_cpn):
				# 			# print('v', cpn)
				# 			cpn_img = trim_image[cpn]
				# 			# show_img('', cpn_img)
				# 			cv2.imwrite('{}{}.{}.png'.format(cut_dir, cell_index, index_word), cpn_img)
				# 			index_word+=1
				# 	else:
				# 		cv2.imwrite('{}{}.{}.png'.format(cut_dir, cell_index, index_word), trim_image)
				# 		index_word += 1
				# else:
					# print('write')
				cv2.imwrite('{}{}.{}.png'.format(cut_dir, cell_index, index_word), trim_image)
				index_word+=1
		# else:
		# 	show_img('{}.{}'.format(cell_index, i), cut_img)

def check_blank_img(img, config):
	# plt.imshow(img)
	# plt.show()
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	invert_img = cv2.bitwise_not(gray)
	norm_img = cv2.normalize(invert_img,None, 0,1, cv2.NORM_MINMAX)
	# #print(norm_img)
	# #print(norm_img)
	thresh_sum = config.getint('checkBlank', 'threshSum')
	sum = np.sum(norm_img)
	#print('sum ', sum)
	if sum< thresh_sum:
		#print('True')
		return True
	else:
		#print('False')
		return False
#lib_img.pkl
import configparser
def words_cut_all(dir):
	'''read image and perform cut cells'''
	#print('dir', dir)
	# cell_file = '25.png'
	# #print('table file', cell_file)
	count = 0
	cut_dir = dir + 'cut'
	config = configparser.ConfigParser()
	path_config = '/home/flax/toni_work/flex_scan_engine2/flex_scan_engine/data/meta/temp3/config8.ini'	
	config.read(path_config)
	if not os.path.exists(cut_dir):
		os.makedirs(cut_dir)
	onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
	for line_file in onlyfiles:
		print(line_file)
	# for line_file in ['mixedline_1.png']:
	# for line_file in ['line_1_1.png']:
	# for line_file in ['01.0001.06.png']:
	# for line_file in ['region_7.png']:
	# for line_file in ['01.0001.08.png']:
		count+=1
		file = '{}{}'.format(dir,line_file)
		# file = '{}{}'.format(dir, table_file)
		#print('line_fine ', line_file)
		word_cut(None, file, line_file, None, cut_dir, config, None)
	#print('count ', count)
dir = '/home/flax/toni_work/flex_scan_engine2/flex_scan_engine/ocrolib/hwocr/gallery/' 
path_config = '/home/flax/toni_work/flex_scan_engine2/flex_scan_engine/data/meta/temp3/config8.ini'

#       'run_form_cut_4_template/03/sougou20017/2/'
words_cut_all(dir)
