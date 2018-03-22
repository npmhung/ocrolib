from matplotlib import pyplot as plt
import numpy as np
import cv2
import logging
from scipy import stats

logging.basicConfig(level = logging.INFO)

def check_dividable_horizontal_cells(img, cell_type, plot):
	print('check dividable')
	hei, wid = img.shape
	print('hei, wid', hei," ", wid)
	hist_x = np.sum(img, axis=0, dtype=np.int32)
	list_bin_x = np.arange(0, wid)
	invert_hist_x = hei - hist_x
	mode = stats.mode(invert_hist_x)[0]
	print('mode ', mode)
	# maximum = np.argmax(invert_hist_x)
	# maximum_pivot = np.argwhere(invert_hist_x> hei/3)
	if cell_type == 'dash':
		thresh_sum = (mode[0]+1)* 7
	maximum_pivot = np.argwhere(invert_hist_x> thresh_sum)
	maximum_pivot = maximum_pivot[(maximum_pivot< wid-20) &(maximum_pivot> 20)]
	check = len(maximum_pivot) > 0
	print('maximum', maximum_pivot)
	'''plot'''
	if plot:
		ax1 = plt.subplot(121)
		ax1.plot(list_bin_x, invert_hist_x)

		ax2 = plt.subplot(122)
		ax2.imshow(img)

		plt.suptitle(str(check))
		plt.tight_layout()
		plt.show()

	if check:
		print('dividable')
		print('wid-20', wid-20)
	else:
		print('not dividable')
	return check

def check_dividable_vertical_cells(img, plot):
	print('check dividable')
	hei, wid = img.shape
	print('hei, wid', hei, " ", wid)
	hist_y = np.sum(img, axis=1, dtype=np.int32)
	list_bin_y = np.arange(0, hei)
	invert_hist_y = wid - hist_y
	maximum = invert_hist_y[invert_hist_y> wid/3]
	maximum_pivot = np.argwhere(invert_hist_y> wid/3)
	maximum_pivot = maximum_pivot[(maximum_pivot < hei-20) & (maximum_pivot> 20)]
	check = len(maximum_pivot) > 0
	print('max pivot', maximum_pivot)
	'''plot'''
	if plot:
		ax1 = plt.subplot(121)
		ax1.set_ylim(0,hei)
		ax1.plot(invert_hist_y, list_bin_y)

		ax2 = plt.subplot(122)
		ax1.set_ylim(0, hei)
		ax2.imshow(img)

		plt.suptitle(str(check))
		plt.tight_layout()
		plt.show()

	if check:
		print('dividable')
	else:
		print('not dividable')
	return check

def mergeCells(img,list_pivot_x, list_pivot_y, cell_type):
	output_pivot_x = np.empty((0),dtype=np.int32)
	for j,y in enumerate(list_pivot_y[:-1]):
		line_pivot_x = list_pivot_x
		for i,x in enumerate(line_pivot_x[:-2]):
			couple_imgs = img[y: list_pivot_y[j+1],x:line_pivot_x[i+2]]
			last_check = check_dividable_horizontal_cells(couple_imgs, cell_type, 0)
			if not last_check:
				print('dividable')
				line_pivot_x = np.delete(line_pivot_x, i+1)
		output_pivot_x = np.hstack((output_pivot_x, line_pivot_x))
	print('output_pivot shape', output_pivot_x.shape)

	output_pivot_y = np.empty((0), dtype=np.int32)
	# for i, x in enumerate(list_pivot_x[:-1]):
	# 	line_pivot_y = list_pivot_y
	# 	for j, y in enumerate(line_pivot_y[:-2]):
	# 		couple_imgs = img[y: list_pivot_y[j + 2], x:line_pivot_x[i + 1]]
	# 		last_check = check_dividable_vertical_cells(couple_imgs, 0)
	# 		if not last_check:
	# 			print('dividable')
	# 			line_pivot_y = np.delete(line_pivot_y, j + 1)
	# 	output_pivot_y = np.hstack((output_pivot_y, line_pivot_y))
	# print('output_pivot shape', output_pivot_x.shape)
	return output_pivot_x, output_pivot_y

def createEdgeMatrix(list_pivots_x, list_pivots_y):
	width_mt, height_mt = len(list_pivots_x)-1, len(list_pivots_y)-1
	print('hei_mt ', 'wid_mt', height_mt, width_mt)
	edge_matrix = np.ones((height_mt, width_mt, 4),dtype=np.int8)
	print('edge matrix', edge_matrix)
	return edge_matrix, width_mt, height_mt

def getContiguousHorizontalCell(img, list_pivot_x, list_pivot_y, x,y):
	print('haha', list_pivot_x[x], list_pivot_y)
	contiguous_cell = img[list_pivot_y[y]:list_pivot_y[y+1],list_pivot_x[x-1]: list_pivot_x[x+1]]
	return contiguous_cell

def getContiguousVerticalCell(img, list_pivot_x, list_pivot_y, x,y):
	contiguous_cell = img[list_pivot_y[y-1]:list_pivot_y[y+1],list_pivot_x[x]: list_pivot_x[x+1]]
	return contiguous_cell

def updateEdgeMatrix(edge_matrix, width_mt, height_mt, list_pivot_x, list_pivot_y, img, cell_type):
	'''update hozirontally'''
	logging.info('update edge horizontal matrix')
	for y in range(0,height_mt):
		for x in range(1,width_mt):
			cell = getContiguousHorizontalCell(img, list_pivot_x, list_pivot_y, x,y)
			plot = 0
			check = check_dividable_horizontal_cells(cell, cell_type, plot)
			'''not devidable'''
			if not check:
				print('yx', y, x)
				edge_matrix[y,x-1][2]= 0
				edge_matrix[y,x][0]= 0

	logging.info('update edge vertical matrix')
	'''update vertically'''
	for x in range(0, width_mt):
		for y in range(1,height_mt):
			cell = getContiguousVerticalCell(img, list_pivot_x, list_pivot_y, x, y)
			plot = 0
			check = check_dividable_vertical_cells(cell, plot)
			print('check', check)
			'''not dividable'''
			if not check:
				print('yx', y,x)
				edge_matrix[y-1,x][3]= 0
				edge_matrix[y,x][1]= 0
	print('updated edge matrix', edge_matrix)
	return edge_matrix

def expand(edge_matrix,start_y, start_x):
	end_y, end_x = start_y, start_x
	next_cell = edge_matrix[end_y, end_x]
	while ((next_cell[2] != 1)):
		print(next_cell[2])
		end_x+=1
		next_cell= edge_matrix[end_y, end_x]
	print('next', next_cell[2])
	while ((next_cell[3] != 1)):
		end_y+=1
		next_cell= edge_matrix[end_y, end_x]
	block = np.array([start_y, end_y+1, start_x, end_x+1], dtype= np.int32)
	print('expand', block)
	return block

def detectBlocks(edge_matrix):
	list_blocks = np.empty([0,4], dtype= np.int32)
	list_top_lefts = np.argwhere((edge_matrix[:,:,:2]== [1,1]).all(axis = 2))
	print('list topleft', len(list_top_lefts))
	for top_left in list_top_lefts:
		list_blocks=  np.row_stack((list_blocks,[expand(edge_matrix, top_left[0], top_left[1])]))
	print('len list block', len(list_blocks), list_blocks)
	return list_blocks

# mergeCells()