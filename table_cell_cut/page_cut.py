# -*- coding: utf-8 -*-
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
sys.path.append('F:/Hellios-workspace/Photoreader/')
dir = "{}Data/Image/".format(sys.path[-1])


def cut_page(filename):
	cut_dir = '{}Data/Image/cut_page/'.format(sys.path[-1])
	fig_dir = '{}Data/Image/cut_page/fig/'.format(sys.path[-1])
	file_path = dir + filename
	img = cv2.imread(file_path)
	hei, wid, c = img.shape
	check_pages = check_num_pages(hei, wid)
	# if check_pages == 2:
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, bin_img = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	print('shape ', bin_img.shape)
	hist = np.sum(bin_img, axis=0)
	list_bin = range(wid)
	trip_hist = hist[int(wid / 3):int(wid * 2 / 3)]
	x_min = np.argmax(trip_hist) + int(wid / 3)
	print('trip ', trip_hist.shape)
	# plt.ylim(ymin =0)
	fig = plt.figure()
	plt.xlim([0, wid])
	plt.plot(x_min, hist[x_min], 'ro')
	plt.plot(list_bin, hist)
	fig.savefig(fig_dir + filename)
	# plt.show()

	print('xmin ', x_min)
	cut_img1 = img[:,0:x_min]
	cut_img2 = img[:,x_min:]
	cv2.line(img, (x_min, 0), (x_min, hei), (0, 0, 255), 8)
	cv2.imwrite(cut_dir+ '1.'+filename, cut_img1)
	cv2.imwrite(cut_dir + '2.' + filename, cut_img2)
	# cv2.imshow('',img)
	cv2.waitKey(0)
	return cut_img1, cut_img2

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
	cut_page(file)
	# cut_page('4.jpg')
