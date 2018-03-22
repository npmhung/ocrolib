from scipy import ndimage
# from ocrolib import morph
# import ocrolib
from numpy import (amax, minimum, maximum, array, zeros, where, transpose)
from ocrolib.preprocess.segmentation import segmentation
from ocrolib.wordcut.wordutils import checkDateField
from ocrolib.preprocess.morphology import enhance
from ocrolib.line_seg_util import normalize_cell_img
from matplotlib import pyplot as plt
import cv2
import numpy as np


def show(title, img):
	plt.imshow(img)
	plt.title(title)
	plt.show()

def cutCpn(im):
	kernel = np.ones((1, 1), np.uint8)
	erose = cv2.erode(im, kernel, iterations=1)
	# show('input', im)
	# im = cv2.normalize(erose, None, 0,1, cv2.NORM_MINMAX)
	# enhanced = enhance(im,None, 1, 15, 2, None, None)
	h, w= im.shape
	# print('shape', h, w)
	label_im, nb_labels = ndimage.label(erose < 120)
	# objects = ndimage.find_objects(label_im)
	# show('label', label_im)
	labels = np.unique(label_im)
	label_im = np.searchsorted(labels, label_im)
	list_cpn = ndimage.find_objects(label_im)
	rm_area = 100
	reserved_list = []
	for i, slice in enumerate(list_cpn):
		'''check if each cpn is mergeable with one another'''
		print('i', i)
		copy_list = list_cpn
		slice_x, slice_y = slice
		print(slice)
		if (slice_x.stop-slice_x.start)*(slice_y.stop- slice_y.start) < rm_area:
			# #print('y')
			continue
		copy_list.pop(i)
		start, stop = slice_y.start,slice_y.stop
		flag = False
		for j, slice_sub in enumerate(copy_list):
			print('j ', j)
			slice_x1, slice_y1 = slice_sub
			'''1.if another one is bigger than current'''
			if slice_y.start >= slice_y1.start and slice_y.stop <= slice_y1.stop:
				print('rm')
				flag = True
				# list_cpn.remove(slice)
				break
			else:
				'''2.current is bigger than another one'''
				if slice_y.start <= slice_y1.start and slice_y.stop >= slice_y1.stop:
					list_cpn.remove(slice_sub)
					# reserved_list.append((slice_y.start, slice_y.stop))
				else:
					'''3.current are overlap with another'''
					sub1, sub2 = slice_y.stop - slice_y1.start,slice_y1.stop - slice_y.start
					# if(sub1 * sub2 < 0): continue
					if sub1> 0:
						if sub2/ sub1 < 10/3:
							start, stop = slice_y.start, slice_y1.stop
					if sub2> 0:
						if sub1 / sub2 < 10/3:
							start, stop = slice_y1.start, slice_y.stop
		if flag: continue
		'''if not 1 then add to reserved'''
		# new_slice = slice(start, stop, None)
		reserved_list.append((start, stop))
		print('leng', len(list_cpn))

	# sorted_cpn = sorted(list_cpn, key=lambda x: x[1].start)
	sorted_cpn = sorted(reserved_list, key=lambda x: x[0])
	len_cpn = len(sorted_cpn)
	if len_cpn == 0:
		return []
	final_cpn = []
	if len_cpn == 1:
		slice_y = sorted_cpn[0]
		final_cpn.append(slice_y)
	else:
		for i in range(len_cpn-1):
			slice_y1 = sorted_cpn[i]
			slice_y2 = sorted_cpn[i+1]
			# slice_y1 = cpn1
			# slice_y2 = cpn2
			gap = slice_y2[0] - slice_y1[1]
			width = slice_y2[1] - slice_y1[0]
			if gap == 0:
				final_cpn.append(slice_y1)
				# show('g', im[:, starty: stopy])
			else:
				if width/gap> 5:
					final_cpn.append(slice_y1)
					# show('', im[starty: stopy])
		last_y = sorted_cpn[len_cpn-1]
		final_cpn.append(last_y)
	return final_cpn

def findComponents(im, filename, line_file, cell_index, cut_dir, config, plot):
	# im = 255 - im
	gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
	kernel = np.ones((1,1), np.uint8)
	erose = cv2.erode(gray, kernel, iterations= 1)
	# show('input', im)
	# im = cv2.normalize(erose, None, 0,1, cv2.NORM_MINMAX)
	h, w, _ = im.shape
	#print('shape', h, w)
	label_im, nb_labels = ndimage.label(erose<120)
	# objects = ndimage.find_objects(label_im)
	# show('label', label_im)
	labels = np.unique(label_im)
	label_im = np.searchsorted(labels, label_im)
	list_cpn = ndimage.find_objects(label_im)
	sorted_cpn = sorted(list_cpn, key= lambda x: x[1].start)
	# isNumber = config.getint('number', 'isnumbercut')
	isDateField = config.getint('datefield', 'isdate')
	if isDateField:
		pivot_d = int(config.getfloat('datefield', 'd') * w)
		pivot_m = int(config.getfloat('datefield', 'm') * w)

	#print('len', nb_labels)
	rm_area = config.getint('connectedcpn', 'remove_area')
	for i, slice in enumerate(sorted_cpn):
		# slice_x, slice_y = ndimage.find_objects(label_im==i)
		slice_x, slice_y = slice
		# #print('st',slice_x.start, slice_x.stop, slice_x.step)
		if slice_x.start == 0 and slice_x.stop == h and slice_y.start== 0 and slice_y.stop== w:
			continue
		if (slice_x.stop-slice_x.start)*(slice_y.stop- slice_y.start) < rm_area:
			# #print('y')
			continue
		if (slice_x.stop-slice_x.start) < 3:
			continue
		if (slice_y.stop-slice_y.start) < 3:
			continue
		#print('slide ',slice_x, slice_y)
		startx = max(slice_x.start - 2, 0)
		stopx = min(slice_x.stop + 2, h-1)
		starty = max(slice_y.start -2, 0)
		stopy = min(slice_y.stop +2, w-1)
		roi = im[startx: stopx, starty: stopy]
		# show('a', roi)
		if isDateField:
			date_index = checkDateField(im, slice_y.start, pivot_d, pivot_m)
			filepath = '{}{}.{}.png'.format(cut_dir, date_index, i)
		else:
			filepath = '{}{}.{}.png'.format(cut_dir,cell_index, i)
		#print('filepath ', filepath)
		roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		roi = normalize_cell_img(roi)
		cv2.imwrite(filepath, roi)


# dir = '/media/warrior/MULTIMEDIA/Newworkspace' \
#       '/Nissay/output/run_form_cut_4_template/03/sougou20005/8/'
# file = 'line_1_1.png'
# filename = dir + file
# img = cv2.imread(filename)
# findComponents(img, 30)
