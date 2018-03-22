from matplotlib import pyplot as plt
import cv2
import numpy as np

def findingNoise(img):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	hei, wid = gray.shape
	ret, thresh = cv2.threshold(gray, 130, 255, 0)
	print(thresh.shape)
	kernel = np.ones((3,3), dtype= np.uint8)
	erode_img = cv2.erode(thresh, kernel, iterations=1)
	# plt.imshow(erode_img)
	# plt.show()
	im2, contours, hierarchy = cv2.findContours(erode_img, cv2.RETR_CCOMP, 2)
	text_coor = np.zeros([0,4], dtype=np.int32)
	for i,cnt in enumerate(contours):
		# compute the center of the contour
		x, y, w, h = cv2.boundingRect(cnt)
		# print('wh ', w*h)
		# print(h/hei)
		# print(x, wid)
		if(w * h < 300) and (y < 15 or x< 10 or (hei - y- h < 15) or (wid- x- w< 10)):
			text_coor = np.row_stack((text_coor,(x,y,w,h)))
				# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
			# remove_cts.append(cnt)
			# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)
	# plt.imshow(img)
	# plt.title('ct')
	# plt.show()
	return text_coor

def denoise(img):
	# print('remove noise')
	remove_ct_coor = findingNoise(img)
	# sort_ct = sorted(remove_ct_coor, key= lambda x: x[0])
	# loop over the contours
	remove_ct_coor = remove_ct_coor.astype(np.int32)
	for c in remove_ct_coor:
		# if the contour is bad, draw it on the mask
		x, y, w, h = c
		img[y+2:y + h -2, x+2:x + w -2] = 255
	return img