import numpy as np
from matplotlib import pyplot as plt
from ocrolib.wordcut.extras import findContour
import cv2


def trimLine(img):
	# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	copy_img = img.copy()
	kernel = np.ones((5,40), np.uint8)
	erosed_img = cv2.morphologyEx(copy_img, cv2.MORPH_RECT, kernel)
	# plt.imshow(erosed_img)
	# plt.show()
	list_coor = findContour(erosed_img)
	hei, wid,_ = img.shape
	len, _ = list_coor.shape
	# print('leng ', len)
	sort_cts = list_coor[list_coor[:, 1].argsort()].astype(np.int32)
	upper = 0
	if len > 0:
		top_ct = sort_cts[0]
		# bottom_ct = sort_cts[0]
		# print('top, bottom ', top_ct)
		upper = max(top_ct[1]-5, 0)
		# lower = top_ct[1]+ top_ct[-1]+3

	lower = img.shape[0]
	lower_cts = list_coor[:, 1] + list_coor[:, -1]
	lower_cts = np.sort(lower_cts).astype(np.int32)
	if len > 0:
		lower = min(lower_cts[-1] +5, hei-1)
	# print('uplow ', upper, lower)
	trimmed_img = img[upper: lower, :]
	# plt.imshow(trimmed_img)
	# plt.show()
	return trimmed_img

def checkDateField(img, pivot_left, pivot_d, pivot_m):
	index = ''
	if pivot_left< pivot_d:
		index = 'd'
	if pivot_left>= pivot_d and pivot_left< pivot_m:
		index = 'm'
	if pivot_left>= pivot_m:
		index = 'y'
	# cp_img = img.copy()
	# hei, wid, _ = img.shape
	# cv2.line(cp_img, (pivot_d, 0), (pivot_d, hei - 1), (255, 255, 0), 2)
	# cv2.line(cp_img, (pivot_m, 0), (pivot_m, hei - 1), (0, 255, 255), 2)
	# cv2.line(cp_img, (pivot_left, 0), (pivot_left, hei - 1), (0, 255, 0), 2)
	# plt.imshow(cp_img)
	# plt.title('%s'%index)
	# plt.show()
	return index

def checkNoise(hei, wid):
	if (hei* wid < 250):
		return True
	else:
		return False