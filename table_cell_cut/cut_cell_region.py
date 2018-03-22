# from src.ocropy.ocrolib.table_cell_cut.cut_cell import pivotDetection, cutCells
import os

import cv2
import numpy as np
from ocrolib.wordcut.wordcutdash import wordcutdash
from ocrolib.wordcut.wordcutct import word_extract
from ocrolib.wordcut.wordcutCntcpn import findComponents

from ocrolib.table_cell_cut.cut_cell_dash import pivotDetection, cutCells
from ocrolib.wordcut.extras import remove_text
from ocrolib.wordcut.wordcuthw import word_cut


def regionCutCell(img, filename, region_file, cell_type, write_dir, field_id, config, plot):
	print('--------------------------Performing word cut -------------------------')
	if img is None:
		img = cv2.imread(filename)
	# plt.imshow(img)
	# plt.title('inputhere')
	# plt.show()
	#print('imgshape ', img.shape)
	hei, wid, _ = img.shape
	# kernel = np.ones((2, 2), np.uint8)
	if cell_type == 'dash':
		dimension = 15
		kernel = np.ones((dimension, int(dimension / 6)), np.uint8)
		erose_line_img = cv2.morphologyEx(img,cv2.MORPH_RECT, kernel)
		img_edge = remove_text(erose_line_img)
		debug = 0
		origin, blank_table, horizontal_arg, vertical_arg = pivotDetection(img_edge, img_edge, debug)

		hei_bl, wid_bl = blank_table.shape
		#print('gray shape', hei, wid, hei_bl, wid_bl)
		#print('listarg ', horizontal_arg, vertical_arg)
		cell_type = 'dash'
		list_coor = cutCells(img, blank_table, horizontal_arg, vertical_arg, wid, hei, cell_type)
		#print('listcoor ', list_coor)
		#copyfile(filename, '{}{}.png'.format(write_cell_dir, region_file[:-4]))
		list_cells = write_cells(img, list_coor,write_dir)
		for i,cell in enumerate(list_cells):
			wordcutdash(cell, None, region_file, i, write_dir, config)
	else:
		if cell_type == 'handwriting':
			#copyfile(filename, '{}{}.png'.format(write_cell_dir, region_file[:-4]))
			word_cut(img, None, region_file, 0, write_dir, config, plot)
		else:
			if cell_type == 'number':
				#print('word number cut')
				#copyfile(filename, '{}{}.png'.format(write_cell_dir, region_file[:-4]))
				# wordCutOptical(img, None, region_file, 0, write_cell_dir, config)
				word_extract(img, None, region_file, 0, write_dir, config)
			else:
				#print('word connected component cut')
				# copyfile(filename, '{}{}.png'.format(write_cell_dir, region_file[:-4]))
				# wordCutOptical(img, None, region_file, 0, write_cell_dir, config)
				findComponents(img, None, region_file, 0, write_dir, config, plot)

def write_cells(img, list_coor, write_cell_dir):
	list_cells = []
	for i, cell in enumerate(list_coor):
		cell_img = img[cell[1]+2:cell[3]-2, cell[0]: cell[2]]
		list_cells.extend([cell_img])
		cv2.imwrite('{}cell_{}.png'.format(write_cell_dir, i), cell_img)
	#print('list_cell ', len(list_cells))
	return list_cells

def cellWordCut(dir, write_dir):
	'''read image and perform cut cells'''
	#print('dir', dir)
	count = 0
	# for line_file in os.listdir(dir):
	# list = ['02.20004.13']
	# list = ['01.0001.06.png']
	# for line_file in ['region_7.png']:
	# list = ['01.0001.08.png']
	# list = ['01.0002.15.png']
	# list = ['02.20005.17.png']
	# list = ['03.0004.21.png']
	# list = ['01.0003.09.png']
	# list = ['03.0002.19.png']
	list = ['02.20008.19.png']
	# list = ['mixedline_1.png']
	# list = ['03.0002.07.png']
	# list = ['04.0032.11.png']
	# list = ['03.0004.22.png']
	# list = ['03.0002.19.png']
	# list = ['02.20005.16.png']
	# list = ['01.0001.15.png']
	# cell_type = 'optical'
	# cell_type = 'handwriting'
	cell_type = 'dash'
	for line_file in list:
		count += 1
		file = '{}{}'.format(dir, line_file)
		#print('fine ', file)
		regionCutCell(None, file, line_file, cell_type, write_dir)
	#print('count ', count)
