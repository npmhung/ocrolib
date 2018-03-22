import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import morphology

''' apply segmentation on image before do connected component'''
# def segmentation(img):
def segmentation(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	# noise removal
	kernel = np.ones((1,1),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

	# sure background area
	sure_bg = cv2.dilate(opening,kernel,iterations=2)
	# plt.imshow(sure_bg)
	# plt.title('sure')
	# plt.show()
	# Finding sure foreground area
	dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
	ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	# plt.imshow(sure_fg)
	# plt.title('surefg')
	# plt.show()
	unknown = cv2.subtract(sure_bg,sure_fg)
	# Marker labelling
	ret, markers = cv2.connectedComponents(sure_fg)

	# Add one to all labels so that sure background is not 0, but 1
	markers = markers+1
	# Now, mark the region of unknown with zero
	markers[unknown==255] = 0
	markers = cv2.watershed(img,markers)
	img[markers == -2] = [255,0,0]
	# cv2.imwrite(dir + 'unkn.png', unknown)
	# cv2.imwrite(dir + 'img.png', img)
	# plt.imshow(unknown)
	# plt.title('unkn')
	# plt.show()
	# plt.imshow(img)
	# plt.title('im')
	# plt.show()
	return img, unknown

def label_component(color_img, img):
	labels = morphology.label(img, background=0)
	print('label', labels)
	# show_image(img)
	label_number = 0
	h, w = img.shape
	# print(h,w)
	while True:
		temp = np.uint8(labels == label_number) * 255
		if not cv2.countNonZero(temp):
			break
		label_number += 1

# dir = '/media/warrior/MULTIMEDIA/Newworkspace' \
#       '/Nissay/output/run_form_cut_4_template/03/sougou20005/8/'
# file = 'line_1_1.png'
# filename = dir + file
# dir = '/media/warrior/MULTIMEDIA/Newworkspace/Nissay/' \
# 	      'output/wordcut/cell_trimming/cellword/01.0002.08/'
# filename = dir  + 'cell_1.png'
# img = cv2.imread(filename)
# color, image = segmentation(img)
# plt.imshow(color)
# plt.title('color')
# plt.show()
# plt.imshow(image)
# plt.title('img')
# plt.show()
# label_component(color, image)