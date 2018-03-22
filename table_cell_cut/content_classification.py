import cv2
import numpy as np
from ocrolib.table_cell_cut. cell_features import numOfLeftCells, colorMean, rightCells, lowerCells


'''table name classify'''
def header1_classify(img, table_name, list_cells):
	img = img*255
	hei, wid = img.shape
	img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	is_head1 = False
	list_left_cells = list_cells[:,1]
	list_top_cells = list_cells[:,0]
	print('list_cells', list_cells)
	condition = ((list_left_cells == 0 ) & (list_top_cells ==0))
	print('condition ', condition)
	candidate_cells = list_cells[condition][0]
	print('candidate', candidate_cells)
	if(len(candidate_cells)==0):
		return is_head1, candidate_cells

	print('candidate_cells ', candidate_cells)
	# sorted_top_cells = np.sort(list_cells, axis=0)
	# num_left_cells = numOfLeftCells(candidate_cells, list_cells)
	print('shape ', img.shape)
	#check whether candidate cells is header 1
	if not ((candidate_cells[2]+1)!= wid)& ((candidate_cells[3]+1)!= hei):
		is_head1 == True
		font = cv2.FONT_ITALIC
		cv2.putText(img, 'header1', (candidate_cells[0], candidate_cells[3]/2), font,2, (0,255,0), 3, cv2.LINE_AA)
	# cv2.imshow('puttext', img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	cv2.imwrite('tb_header{}.png'.format(table_name), img)
	print('is_head ', is_head1)
	return is_head1, candidate_cells


def header2_classify(gray_img, list_coordinates_cells, list_cell_imgs):
	is_head2 = False
	list_cell_imgs = np.array([gray_img[cell[1]:cell[3], cell[0]: cell[2]] for cell in list_coordinates_cells])
	list_candidate_cells = list_cell_imgs[np.mean(list_cell_imgs)<200]
	for cell in list_candidate_cells:
		right_cells = rightCells(cell, list_coordinates_cells)
	print('list candidate cells', list_candidate_cells)
	color_feats = colorMean(gray_img)

	# if(color_feats< 200):

	return is_head2


def header3_classify():
	is_head3 = False

	return is_head3


def header4_classify():
	is_head4 = False

	return is_head4


def value1_classify():
	is_value1 = False

	return is_value1

def value2_classify():
	is_value2 = False

	return is_value2