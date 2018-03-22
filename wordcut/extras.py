import cv2
from matplotlib import pyplot as plt
import numpy as np

def findContour(img):
	# plt.imshow(img)
	# plt.show()
	hei_line, wid_line, _ = img.shape
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	border_img = cv2.copyMakeBorder(gray, 1,1,1,1, cv2.BORDER_CONSTANT, value= [255,255,255])
	ret, thresh = cv2.threshold(border_img, 120, 255, 0)
	im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, 2)
	list_coor = np.zeros([0,4])
	for i,cnt in enumerate(contours):
		# compute the center of the contour
		x, y, w, h = cv2.boundingRect(cnt)
		if (x<5 or wid_line - x < 5)and w< 5:
			continue
		if (y<5 or hei_line - y < 5)and h<5:
			continue
		if h/ hei_line > 0.95 and w/ wid_line > 0.95:
			continue
		if h == hei_line and w < 5:
			continue
		if w == wid_line and h< 5:
			continue
		list_coor = np.row_stack((list_coor,(x,y,w,h)))
		# print('wh ', w*h)
		# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
	# plt.imshow(img)
	# plt.title('ct')
	# plt.show()
	return list_coor

def count_contour(img, erosed_img):
	# plt.imshow(img)
	# plt.show()
	# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	ret, thresh = cv2.threshold(img, 120, 255, 0)
	im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, 2)
	check_num_ct = 0
	for i,cnt in enumerate(contours):
		# compute the center of the contour
		x, y, w, h = cv2.boundingRect(cnt)
		# print('wh ', w*h)
		if(w * h >120):
			check_num_ct+=1
		# cv2.circle(img, (cX, cY), 3, (255, 0, 255), -1)
		# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
	# plt.imshow(img)
	# plt.title('ct')
	# plt.show()
	# plt.imshow(erosed_img)
	# plt.show()
	# print('check num ct ', check_num_ct)
	return check_num_ct

def finding_contour(img, select):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	hei, wid = gray.shape
	ret, thresh = cv2.threshold(gray, 130, 255, 0)
	print(thresh.shape)
	im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, 2)
	check_num_ct = 0
	remove_ct_coor = np.zeros([0,4], dtype=np.int32)
	for i,cnt in enumerate(contours):
		# compute the center of the contour
		x, y, w, h = cv2.boundingRect(cnt)
		if select:
			if(w * h < 50) or ((abs(x-wid)<6 or x<=6) and h/hei>0.9 and w<3):
				# print('remove dash')
				check_num_ct+=1
				remove_ct_coor = np.row_stack((remove_ct_coor,(x,y,w,h)))
				# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
		else:
			if not ((w * h < 30) or ((abs(x-wid)<10 or x< 10) and h/hei>0.9)):
				# print('remove_text')
				check_num_ct += 1
				remove_ct_coor = np.row_stack((remove_ct_coor, (x, y, w, h)))
			# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)
			# remove_cts.append(cnt)
		# else:
		# 	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)
	# plt.imshow(img)
	# plt.title('ct')
	# plt.show()
	return remove_ct_coor

def remove_dash(img):
	# img = cv2.imread('/media/warrior/MULTIMEDIA/Newworkspace/Nissay/output/test/1.png')
	# plt.imshow(img)
	# plt.title('dash')
	# plt.show()
	kernel = np.ones((2, 2), np.uint8)
	erosion = cv2.morphologyEx(img, cv2.MORPH_RECT, kernel)
	# plt.imshow(erosion)
	# plt.show()
	select = True
	remove_ct_coor= finding_contour(erosion, select)
	# sort_ct = sorted(remove_ct_coor, key= lambda x: x[0])
	# loop over the contours
	remove_ct_coor= remove_ct_coor.astype(np.int32)
	for c in remove_ct_coor:
		# if the contour is bad, draw it on the mask
		x,y, w, h = c
		img[y:y+h, x:x+w] = 255
		# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
	# plt.imshow(img)
	# plt.show()
	return img

def remove_text(img):
	removed_text_img = img.copy()
	# plt.imshow(removed_text_img)
	# plt.title('before')
	# plt.show()
	select = False
	remove_ct_coor = finding_contour(removed_text_img, select)
	# loop over the contours
	remove_ct_coor = remove_ct_coor.astype(np.int32)
	for c in remove_ct_coor:
		# if the contour is bad, draw it on the mask
		x, y, w, h = c
		removed_text_img[y:y + h, x:x + w] = 255
	# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
	# plt.imshow(removed_text_img)
	# plt.title('rm text')
	# plt.show()
	return removed_text_img

def findingText(img):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	hei, wid = gray.shape
	ret, thresh = cv2.threshold(gray, 130, 255, 0)
	print(thresh.shape)
	im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, 2)
	check_num_ct = 0
	text_coor = np.zeros([0,4], dtype=np.int32)
	for i,cnt in enumerate(contours):
		# compute the center of the contour
		x, y, w, h = cv2.boundingRect(cnt)
		# print('wh ', w*h)
		# print(h/hei)
		# print(x, wid)
		if(w * h > 50):
			print('remove dash')
			check_num_ct+=1
			text_coor = np.row_stack((text_coor,(x,y,w,h)))
				# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
			# remove_cts.append(cnt)
			# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)
	# plt.imshow(img)
	# plt.title('ct')
	# plt.show()
	return text_coor

# file = '01.0001.06.png'
# dir = '/media/warrior/MULTIMEDIA/Newworkspace/Nissay/output/wordcut/one_line/'
# img = cv2.imread('{}{}'.format(dir, file))
# print(img.shape)
# remove_text(img)

