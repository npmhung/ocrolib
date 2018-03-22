import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import pkg_resources

DATA_FOLDER = pkg_resources.resource_filename(__name__,'examples')
WINDOW_SIZE = [32,32] #Width and Height of the Sliding Windows
MAX_SEQ_LEN = 48
		

def Load_Image_List(folder):
	print('Load Images from Folder: "%s"' % (folder))
	img_list = []
	file_list = sorted(os.listdir(folder))
	for file_name in file_list:
		img = mpimg.imread(os.path.join(folder, file_name))
		img = cv2.GaussianBlur(img,(5,5),0)
		img = (img[:,:,0]<0.92).astype(float)
		img = cv2.resize(img, (img.shape[1]*WINDOW_SIZE[1]//img.shape[0],WINDOW_SIZE[1]))
		img_list.append(Processing(img))
	
	print(' --> Number of images:',len(img_list))
	return np.asarray(img_list)

def Centering(img, expected_size):
	from scipy import ndimage
	centroid = ndimage.measurements.center_of_mass(img)
	if np.isnan(centroid[0]): return img
	
	shift_x = (expected_size[1] - img.shape[1])//2
	shift_y = expected_size[0]//2 - int(np.ceil(centroid[0]))
	#print(shift_y, img[-shift_y:, :].shape)
	new_img = np.zeros(expected_size)
	if shift_y  == 0: new_img[:,shift_x:shift_x+img.shape[1]] = img[:,:]
	elif shift_y > 0: new_img[shift_y:,shift_x:shift_x+img.shape[1]] = img[:-shift_y,:]
	elif shift_y < 0: new_img[:shift_y,shift_x:shift_x+img.shape[1]] = img[-shift_y:, :]
	
	return new_img

from matplotlib import pyplot as plt
def Processing(img):
	opencv_img = np.uint8(img*255)
	edge = cv2.Canny(opencv_img,50,150)/255
	dx = cv2.Scharr(opencv_img,ddepth=-1,dx=1,dy=0)/255
	dy = cv2.Scharr(opencv_img,ddepth=-1,dx=0,dy=1)/255
		
	aggregated_features = np.stack([img, edge, dx, dy], axis=-1)
	return aggregated_features

def Cutting(img, max_len):
	w,h = WINDOW_SIZE
	img_pieces = []
	"""
	i = 0
	while i < 853-32:
		img_pieces.append(img[:,i:i+w])
		i+=w//2
	img_pieces = [Processing(piece) for piece in img_pieces]
	l = len(img_pieces)
	img_pieces.extend([np.zeros([4,h,w]) for _ in range(max_len-l)])
	return (np.asarray(img_pieces), l+1)
	"""
	
	### Moving a pointer through all columns of the image ###
	shortest_stroke_width = 5
	p = previous_p = 0
	while p < (img.shape[1]-w):
		value = sum(img[:, p])
		if value < 1:
			p += 1; continue
		
		### When seeing a non-blank column --> Find the next blank column ###
		if p - previous_p > 20: img_pieces.append(np.zeros([h,w]))
		#if previous_p > 0 and p - previous_p > 20: img_pieces.append(np.zeros([h,w]))			
		k = p+1
		while sum(img[:, k]) > 0: k += 1
		
		width = k-p
		if width > shortest_stroke_width: 
			if width <= w: 
				img_pieces.extend([Centering(img[:, p:k], [h,w]),img[:,p:p+w]])
			elif width <= w*2: 
				#img_pieces.extend([img[:, p:p+w], img[:, k-w:k]])
				m = p+width//2 # Middle point
				img_pieces.extend([img[:, p:p+w], img[:, m-w//2:m+w//2], img[:, k-w:k]])
			elif width <= w*3:
				m = p+width//2 # Middle point
				#img_pieces.extend([img[:, p:p+w], img[:, m-w//2:m+w//2], img[:, k-w:k]])
				img_pieces.extend([img[:, p:p+w], img[:,m-w:m], img[:, m-w//2:m+w//2], 
												  img[:,m:m+w], img[:, k-w:k]])
		previous_p = p = k
			
	l = len(img_pieces)
	img_pieces = [Processing(Centering(piece,[h,w])) for piece in img_pieces]
	### Padding to match the required tensor shape of TensorFlow ###
	img_pieces.extend([np.zeros([4,h,w]) for _ in range(max_len-l)])
	return (np.asarray(img_pieces), l)
	
	
""" Label loading and vectorizing """
def Temp_SmallVectorization(addr_list):
	char_list = []
	for addr in addr_list:
		for code in addr: 
			if not (code in char_list): char_list.append(code)
	char_list = sorted(char_list)
	for i in range(len(addr_list)):
		for j in range(len(addr_list[i])):
			addr_list[i][j] = char_list.index(addr_list[i][j])
	return char_list

def Load_Label_List(file_path, n_samples):
	print('Load Labels from file: "%s"' %(file_path))
	encoded_addr_list = []
	with open(file_path, 'rt', encoding='utf-8') as f:
		for text_line in f.readlines()[:n_samples]:
			encoded_addr = []
			for character in text_line.strip():
				code = ord(character) # Take the Unicode (decimal) value of char				
				encoded_addr.append(code)
			encoded_addr_list.append(encoded_addr)
			#encoded_addr_list.append(np.asarray(encoded_addr))
	print(' --> Number of labels:',len(encoded_addr_list))
	
	#VectorizeByUnicode(code)
	char_list = Temp_SmallVectorization(encoded_addr_list)
	print(' --> Number of characters:', len(char_list))
	return encoded_addr_list, char_list

def Labels_To_Sparse_Tuple(batch, dtype=np.int32):
	""" Ref: https://github.com/igormq/ctc_tensorflow_example """
	"""Create a sparse representention of x.
	Args: sequences: a list of lists of type dtype where each element is a sequence
	Returns: A tuple with (indices, values, shape)
	"""
	indices = []
	values = []
	for iseq, seq in enumerate(batch):
		indices.extend(zip([iseq]*len(seq), range(len(seq))))
		values.extend(seq)
	indices = np.asarray(indices, dtype=np.int64)
	values = np.asarray(values, dtype=dtype)
	shape = np.asarray([len(batch), indices.max(0)[1]+1], dtype=np.int64)

	return (indices, values, shape)

def Sparse_Tuple_To_Labels(x):
	bs = x[2][0]
	batch = [[] for _ in range(bs)]
	for i in range(len(x[0])):
		seq_id, pos = x[0][i]
		batch[seq_id].append(x[1][i])
	return batch

def Show_Label_Batch(lbls):
	for lbl in lbls:
		print('	', ''.join([chr(char_list[c]) for c in lbl]))

""" Load data into Memory"""
images = Load_Image_List(DATA_FOLDER+'/images-handwritten')
#processed_imgs = [Cutting(img, MAX_SEQ_LEN) for img in images]
n_samples = len(images)
labels, char_list = Load_Label_List(DATA_FOLDER+'/japanese_address.txt', n_samples)

def Next_Batch(batch_size):
    indices = np.random.permutation(n_samples)[:batch_size]
    batch_imgs = [images[i] for i in indices]
    batch_labels = [labels[i] for i in indices]
    return (np.asarray(batch_imgs), batch_labels)


if __name__ == '__main__':
	a = 1
	
