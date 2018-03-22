import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
from scipy import ndimage

def show(title, img):
	plt.imshow(img)
	plt.title(title)
	plt.show()

def enhance(img, filename, orientation, dimension, hor_coef, ver_coef, config):
	if img is None:
		img = cv2.imread(filename)
	if (len(img.shape) > 2):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret, thresh = cv2.threshold(gray, 120, 255, 0)
	else:
		ret, thresh = cv2.threshold(img, 120, 255, 0)
	if config != None:
		hor_coef = config.getfloat('enhance','horCoef')
		ver_coef = config.getfloat('enhance','verCoef')
	if orientation == 0:
		# dimension = 30
		# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (dimension, dimension))
		kernel = np.ones((int(dimension/ver_coef), dimension), np.uint8)
	else:
		# dimension = 10
		# kernel = np.ones((dimension, int(dimension/2)), np.uint8)
		kernel = np.ones((dimension, int(dimension / hor_coef)), np.uint8)
	# cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
	# cv2.imshow('gradient', gradient)
	# cv2.waitKey(0)
	erosion = cv2.morphologyEx(thresh, cv2.MORPH_RECT, kernel)

	return erosion

def enhance_text(img):
	gray = img
	if (len(img.shape) > 2):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret, thresh = cv2.threshold(gray, 120, 255, 0)
	kernel = np.array([[1,1,1,1],
	                   [1,15,15,1],
	                   [1,15,15,1],
	                  [1,1,1,1]])
	erosion = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel)
	plt.imshow(erosion)
	plt.show()
	return erosion

def fill_hole(img, config):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# show('',gray)
	ret, thresh = cv2.threshold(gray, 120, 255, 0)
	norm = cv2.normalize(thresh, None,0,1, cv2.NORM_MINMAX)
	invert = cv2.bitwise_not(norm)
	show('invert', invert)
	fill = ndimage.binary_fill_holes(invert)
	show('fill', fill)
	# dimension = 10
	# # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (dimension, dimension))
	# kernel = np.ones((int(dimension), dimension), np.uint8)
	# # cv2.imshow('gradient', gradient)
	# # cv2.waitKey(0)
	# erosion = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	# show('erose',erosion)
	# cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))

# dir = '/media/warrior/MULTIMEDIA/Newworkspace/Nissay/output/' \
#       'run_form_cut_4_template/03/sougou20001/3/line_1_1.png'
# img = cv2.imread(dir)
# fill_hole(img, None)

def smoothen(img):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	kernel = np.ones((2, 2), np.uint8)
	dilate = cv2.erode(gray, kernel, iterations=1)
	# show('dilate', dilate)
	return dilate