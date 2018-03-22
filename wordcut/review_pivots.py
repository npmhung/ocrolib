import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

def show(title, img):
	plt.imshow(img)
	plt.title(title)
	plt.show()

def find_cpn(im, config):
	if len(im.shape)> 2:
		gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
	else:
		gray = im
	# show('input', im)
	# print('shape', h, w)
	label_im, nb_labels = ndimage.label(gray < 120)
	# show('label', label_im)
	labels = np.unique(label_im)
	label_im = np.searchsorted(labels, label_im)
	list_cpn = ndimage.find_objects(label_im)
	# sorted_cpn = sorted(list_cpn, key=lambda x: x[1].start)
	return list_cpn
# print('len', nb_labels)

def reviewPivots(im, list_cpn, pivot, config):
	'''check whether cut into text'''
	recheck_area = config.getint('reduction', 'recheck_area')
	thresh_review = config.getint('reduction', 'threshmid')
	h, w, _ = im.shape
	list_check = np.empty([0])
	for i, slice in enumerate(list_cpn):
		slice_x, slice_y = slice
		# #print('st',slice_x.start, slice_x.stop, slice_x.step)
		if slice_x.start == 0 and slice_x.stop == h and slice_y.start== 0 and slice_y.stop== w:
			continue
		if (slice_x.stop-slice_x.start)*(slice_y.stop- slice_y.start) < recheck_area:
			continue
		if (slice_x.stop-slice_x.start) < 3:
			continue
		if (slice_y.stop-slice_y.start) < 3:
			continue
		# cv2.line(im,(pivot, 0),(pivot, h-1), (255,255,0), 1)
		# roi = im[slice_x,slice_y]
		# show('', roi)
		# print('sl ', slice_x, slice_y)
		# print('mid',(slice_y.stop + slice_y.start) / 2)
		if abs(pivot- (slice_y.stop+ slice_y.start)/2) < thresh_review:
			list_check = np.append(list_check, True)
		else:
			list_check = np.append(list_check, False)
	check = np.any(list_check)
	# print('check', check)
	return check
		# list_text = np.row_stack((list_text, slice_x.start, slice_x.stop, slice_y.start, slice_y.stop))

