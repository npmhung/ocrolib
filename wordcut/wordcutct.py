import cv2
import os
from matplotlib import pyplot as plt
from ocrolib.preprocess.morphology import enhance, smoothen


def check_box_inside_box(box1, box2):
	'''box1 inside box2'''
	# #print('box1 ', box1, ' box2 ', box2)
	'''0:x, 1:y, 2:w, 3:h'''
	if (box1[1]>=box2[1]) and box1[3] <= box2[3] and (box1[0]+ box1[2]) <= (box2[0]+ box2[2]) and (box1[1]+ box1[1]) <= (box2[1]+ box2[1]):
		return True
	return False

def get_final_box(boxes):
	final_kept_boxs = []
	length = len(boxes)
	#print('box ', boxes)
	for i, box in enumerate(boxes):
		# j=i+1
		# final_kept_boxs.extend(box)
		rm_boxes = boxes
		rm_boxes.pop(i)
		if len(rm_boxes) == 0 : break
		for j, box_check in enumerate(rm_boxes):
			if check_box_inside_box(box, box_check):
				boxes = rm_boxes

		# if i >= length-1: break
	#print('len_final_boxs', len(boxes))
	return boxes

def word_extract(img, filename, line_file, cell_index, cut_dir, config):
	# save_dir = '/media/warrior/MULTIMEDIA/Newworkspace/Nissay/output/wordcut/'
	# img = cv2.imread(line_file)
	# cv2.imshow('img', img)
	# cv2.waitKey(0)
	# plt.imshow(img)
	# plt.title('wordct')
	# plt.show()
	#print('shape', img.shape)
	# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	dimension_kernel = 12
	# gray = enhance(img,filename, 1, dimension_kernel)
	gray = smoothen(img)
	ret,thresh = cv2.threshold(gray,120,255,0)
	im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP , 2)

	'''edge canny instead of threshold'''
	height, width, _ = img.shape
	idx = 0
	#print(len(contours))
	def get_contour_precedence(contour, cols):
		tolerance_factor = 10
		origin = cv2.boundingRect(contour)
		return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

	# contours.sort(key=lambda x:get_contour_precedence(x, img.shape[1]))
	kept_contours = []
	for i, cnt in enumerate(contours):
		x, y, w, h = cv2.boundingRect(cnt)
		kept_contours.append((x, y, w, h))

	kept_contours.sort(key=lambda x: x[0])
	final_boxes = get_final_box(kept_contours)
	# final_boxes = kept_contours
	for i,cnt in enumerate(final_boxes):
		x, y, w, h = cnt
		'''this is for running a format'''
		#print('x ', x, y, w, h)
		# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
		cut_img = img[y:y+h, x:x+w]
		filepath = '{}{}.{}.png'.format(cut_dir, cell_index, i)
		# #print('filepath ', filepath)
		cv2.imwrite(filepath, cut_img)
		idx += 1
		#print('idx ', idx)

	# plt.imshow(img)
	# plt.show()
	# plt.imsave('{}.png'.format(save_dir, filename), img)

# dir = '/media/warrior/DATA/Hellios-workspace/Photoreader/Data/output/1/cut/2/'
# file = '1.png'
# dir = '/media/warrior/MULTIMEDIA/Newworkspace/Nissay/output/02/02/'
# file = '/table02_cell1a_line0002.png'
# dir = '/home/warrior/Project/Photoreader/data/newdata/'
# dir = '/media/warrior/MULTIMEDIA/Newworkspace' \
#       '/Nissay/output/run_form_cut_4_template/03/sougou20004/3/'
# file = 'line_1_1.png'
# filename = dir  + file
# # file = '0.png'
# # filename = '{}{}'.format(dir, file)
#
# word_extract(filename, file)