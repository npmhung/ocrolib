# -*- coding: utf-8 -*-
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
sys.path.append('F:/Hellios-workspace/Photoreader/')
dir = "{}Data/Image/cut_page/".format(sys.path[-1])

def morphology(img):
	kernel = np.ones((10, 15), np.uint8)
	# cross = cv2.morphologyEx(img, cv2.MORPH_ERODE,kernel)
	_,thresh  = cv2.threshold(img, 0, 250, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
	cross = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel)
	cv2.namedWindow("cross", cv2.WINDOW_NORMAL)
	cv2.resizeWindow('cross',800,600)
	cv2.imshow('cross', cross)
	# cv2.resizeWindow('cross',500,400)
	cv2.waitKey(0)


def cut_page(filename):
	cut_dir = '{}Data/Image/cut_page/'.format(sys.path[-1])
	fig_dir = '{}Data/Image/cut_page/fig/vertical_fig/'.format(sys.path[-1])
	file_path = dir + filename
	img = cv2.imread(file_path)
	hei, wid, c = img.shape
	check_pages = check_num_pages(hei, wid)
	# if check_pages == 2:
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	morpho = morphology(gray)
	_, bin_img = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	print('shape ', bin_img.shape)
	hist = np.sum(bin_img, axis=1, dtype= np.int32)
	list_bin = range(hei)
	diff = np.diff(hist)
	list_bin_diff = range(hei-1)
	# trip_hist = hist[int(wid / 3):int(wid * 2 / 3)]
	# x_min = np.argmax(trip_hist) + int(wid / 3)
	# print('trip ', trip_hist.shape)
	# plt.ylim(ymin =0)
	fig = plt.figure()
	ax1 = plt.subplot(221)
	ax1.set_xlim([0,wid])
	ax1.set_ylim([hei, 0])
	# plt.plot(hist[x_min], list_bin[x_min], 'ro')
	plt.plot(hist, list_bin)

	ax2 = plt.subplot(222)
	ax2.set_xlim([0, wid])
	ax2.set_ylim([hei,0])
	plt.imshow(img)

	ax3 = plt.subplot(212)
	ax3.set_xlim([0, wid])
	ax3.set_ylim([hei,0])
	plt.plot(diff, list_bin_diff)
	plt.ylim([hei,0])
	plt.tight_layout()
	# fig.savefig(fig_dir + filename[:-4]+ '.png', dpi = 1200)
	plt.show()

	# print('xmin ', x_min)
	# cut_img1 = img[:,0:x_min]
	# cut_img2 = img[:,x_min:]
	# cv2.line(img, (x_min, 0), (x_min, hei), (0, 0, 255), 8)
	# cv2.imwrite(cut_dir+ '1.'+filename, cut_img1)
	# cv2.imwrite(cut_dir + '2.' + filename, cut_img2)
	# cv2.imshow('',img)
	cv2.waitKey(0)

def check_num_pages(hei, wid):
	coef = 1.5
	if 2 * hei / wid > coef:
		page_type = 1
	else:
		page_type = 2
	print('page_type', page_type)
	return page_type


for file in os.listdir(dir):
	print('file ', file)
	if os.path.isfile(dir+file):
		cut_page(file)
	# cut_page('4.jpg')
