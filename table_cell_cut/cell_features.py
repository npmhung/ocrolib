import cv2
import numpy as np

#cell = np.array(top, left,, bottom, right)
# def sorted_left(list_cells, axis):
# 	return np.sort(list_cells,axis=1)

'''0: left, 1: upper, 2: right, 3: lower'''

# count number of left cells on a cell
def numOfLeftCells(cell, list_cells):
	print(list_cells)
	list_right_cells = list_cells[:,2]
	select_left_cells = list_cells[list_right_cells == cell[0]]
	print('select ', select_left_cells)
	return len(select_left_cells)

def colorMean(img):
	return np.mean(img)

def rightCells(cell, list_cells):
	print(list_cells)
	list_left_cells = list_cells[:,0]
	select_right_cells = list_cells[(list_left_cells == cell[2])]
	value_right_cell = select_right_cells[(select_right_cells> cell[1])&(select_right_cells< cell[3])]
	print('select ', select_right_cells)
	print('select ', value_right_cell)
	return len(select_right_cells)

def lowerCells(cell, list_cells):
	print(list_cells)
	list_lower_cells = list_cells[:,1]
	select_lower_cells = list_cells[(list_lower_cells == cell[2])]
	value_lower_cell = select_lower_cells[(select_lower_cells> cell[0])&(select_lower_cells< cell[2])]
	print('select ', select_lower_cells)
	print('select ', value_lower_cell)
	return len(select_lower_cells)