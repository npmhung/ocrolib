import cv2
from scipy import misc

def rotate(img, config):
	# file = 'HR_B_01_03_ス.jpg'
	# img = misc.imread('E:/Hellios-workspace/Japanese_characters/data/Katakana/'+ file)
	top = config.getint('rotation','top')
	bottom = config.getint('rotation', 'bottom')
	left = config.getint('rotation','left')
	right = config.getint('rotation','right')
	# top,bottom, left, right = 30,30,5, 5
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	img_border = cv2.copyMakeBorder(gray, top, bottom, left, right, cv2.BORDER_CONSTANT,
	                                value = [255])
	invert = cv2.bitwise_not(img_border)


	angle = config.getint('rotation', 'angle')
	# angle = 10
	rotate = misc.imrotate(invert, angle)
	invert_rotate = cv2.bitwise_not(rotate)
	invert_rotate = cv2.cvtColor(invert_rotate, cv2.COLOR_GRAY2RGB)
	# misc.imsave('E:/Hellios-workspace/Japanese_characters/data/Katakana/'+ file, rotate)
	return invert_rotate