# This code is based on
# https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
# https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
import scipy.misc
from keras import backend as K
from keras import initializers
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import optimizers
from sklearn.model_selection import train_test_split
from six.moves import cPickle
from matplotlib import pyplot as plt


nb_classes = 881
# input image dimensions
img_rows, img_cols = 64, 64
# num_writers = 1411
num_writers = 5

def make_data(kind_script, index):
	# img_rows, img_cols = 127, 128
	# kary = np.load(kind_script + ".npz")
	# print('sp ', type(kary))
	# print('sp')
	height, width = 64,63
	#height, width = 127, 128
	dir = './data/'
	ary = np.load(dir+kind_script+ str(index) + ".npz")['arr_0']
	print(ary.shape)
	ary = ary.reshape([-1, height, width]).astype(np.float32) / 15
	# ary = cPickle.load(file)/15
	print('vb', ary.shape)
	# ary = cPickle.load(file).reshape(astype(np.float32) / 15
	X_train = np.zeros([nb_classes * num_writers, img_rows, img_cols])
	for i in range(nb_classes * num_writers):
		X_train[i] = scipy.misc.imresize(ary[i], (img_rows, img_cols), mode='F')
		# X_train[i] = ary[i]
	Y_train = np.repeat(np.arange(nb_classes), num_writers)

	X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)

	if K.image_dim_ordering() == 'th':
		print('xx')
		X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
		X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
		input_shape = (1, img_rows, img_cols)
	else:
		print('yy')
		X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
		X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
		input_shape = (img_rows, img_cols, 1)

	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(Y_train, nb_classes)
	Y_test = np_utils.to_categorical(Y_test, nb_classes)

	datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.20)
	datagen.fit(X_train)
	return X_train, Y_train, X_test, Y_test, datagen

def my_init(shape, dtype=None):
	return K.random_normal(shape, stddev= 0.1, dtype=dtype)

def init_model(model_name,script):
	input_shape = (img_rows, img_cols, 1)
	model = Sequential()
	if model_name == 'm6_1':
		m6_1(model, input_shape)

	if model_name== 'm3':
		m3(model,input_shape)

	model_json = model.to_json()
	with open("./tmp/"+model_name+"_"+script+"_model.json", "w") as json_file:
			json_file.write(model_json)

	return model
	#training(model,model_name,script)

# Best val_loss: 0.0205 - val_acc: 0.9978 (just tried only once)
# 30 minutes on Amazon EC2 g2.2xlarge (NVIDIA GRID K520)
def m3(model, input_shape):
	model.add(Dense(5000, init=my_init, input_shape= input_shape))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(5000, init=my_init))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(5000, init=my_init))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

def m6_1(model, input_shape):
	model.add(Convolution2D(32, 3, 3, init=my_init, input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3, init=my_init))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Convolution2D(64, 3, 3, init=my_init))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, init=my_init))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(256, init=my_init))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

def m6_2(model, input_shape):
	model.add(Convolution2D(64, 3, 3, init=my_init, input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Convolution2D(128, 3, 3, init=my_init))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Convolution2D(512, 3, 3, init=my_init))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(4096, init=my_init))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, init=my_init))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

def m12(model, input_shape):
	model.add(Convolution2D(64, 3, 3, init=my_init, input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, init=my_init, input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Convolution2D(128, 3, 3, init=my_init))
	model.add(Activation('relu'))
	model.add(Convolution2D(128, 3, 3, init=my_init))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Convolution2D(256, 3, 3, init=my_init))
	model.add(Activation('relu'))
	model.add(Convolution2D(256, 3, 3, init=my_init))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Convolution2D(512, 3, 3, init=my_init))
	model.add(Activation('relu'))
	model.add(Convolution2D(512, 3, 3, init=my_init))
	model.add(Activation('relu'))
	model.add(Convolution2D(512, 3, 3, init=my_init))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(4096, init=my_init))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, init=my_init))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

def classic_neural(model, input_shape):
	model.add(Flatten(input_shape=input_shape))
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

def training(model, model_name, script):
	model.summary()
	optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
	# optimizer = optimizers.Adam(lr=1.0, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['accuracy'])
	filepath = "./save/"+ model_name+"_"+ script+ "_weights.best.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	num_subdataset = 1000
	for e in range(1,num_subdataset+1):
		print('dataset'+str(e))
		X_train, Y_train, X_test, Y_test, datagen = make_data(script,1)
		model.fit_generator(datagen.flow(X_train, Y_train, batch_size=16), samples_per_epoch=X_train.shape[0],
							nb_epoch=10, callbacks= callbacks_list, validation_data=(X_test, Y_test))
		model.save_weights("./tmp/"+ model_name+ "_" + script +".h5")

def check_data():
	# np.load('kanji')
	# kary = np.load('E:/Hellios-workspace/Japanese_characters/v.npz')
	# print('kary ', kary['arr_0'].shape)
	file = open('E:/Hellios-workspace/Japanese_characters/CnnJapaneseCharacter/src/kanji/kanji1.pkl', 'rb')
	data = cPickle.load(file)
	print('ss ', data[0].shape)
	plt.imshow(data[0,1])
	plt.show()
	plt.imshow(data[0, 2])
	plt.show()
	plt.imshow(data[0, 3])
	plt.show()


if __name__=='__main__':
	# model = init_model('kanji')
	# model_name = 'm3'
	model_name = 'm6_1'
	script = 'kata'
	model = init_model(model_name, script)
	training(model,model_name,script)
	# classic_neural()
	# check_data()

