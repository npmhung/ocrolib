# This code is based on
# https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
# https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

import numpy as np
import scipy.misc
from keras import backend as K
from keras import initializers, optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import os
import cv2

nb_classes = 884 #72
# input image dimensions
img_rows, img_cols = 32, 32
num_person = 160 #160
# img_rows, img_cols = 127, 128

def load_image(dir):
    im_names = []
    arr_img = np.empty([0,img_rows, img_cols,1])
    for file in sorted(os.listdir(dir)):
        print('file ', file)
        im_names.append(file)
        filename = dir + file
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE )
        img_resize = cv2.resize(img,(32,32))
        out = cv2.normalize(img_resize.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        print('shape ', img_resize.shape, ' content', out)
        reshape = out.reshape([-1,img_rows, img_cols,1])
        arr_img = np.row_stack((arr_img, reshape))
    #print('arr ', arr_img.shape)
    return arr_img, im_names

# ary = np.load("/home/ubuntu/japan_hw/kanji.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32) / 15
# X_train = np.zeros([nb_classes * num_person, img_rows, img_cols], dtype=np.float32)
# for i in range(nb_classes * num_person):
#     X_train[i] = scipy.misc.imresize(ary[i], (img_rows, img_cols), mode='F')
#     # X_train[i] = ary[i]
# Y_train = np.repeat(np.arange(nb_classes), num_person)
#
# X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)
#
# if K.image_dim_ordering() == 'th':
#     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
#     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
#
# # convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(Y_train, nb_classes)
# Y_test = np_utils.to_categorical(Y_test, nb_classes)
#
# datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.20)
# datagen.fit(X_train)

model = Sequential()

def my_init(shape, dtype=None):
    return K.random_normal(shape, stddev= 0.1, dtype=dtype)
input_shape = (img_rows, img_cols, 1)
# Best val_loss: 0.0205 - val_acc: 0.9978 (just tried only once)
# 30 minutes on Amazon EC2 g2.2xlarge (NVIDIA GRID K520)
def m6_1():
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


def classic_neural():
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

def test():
    m6_1()
    # classic_neural()

    filepath = "./checkpoints_server/weights_m61.best.hdf5.latest_98_6"

    # model.summary()

    # resume from checkpoint
    model.load_weights(filepath)

    #optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    # optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06, decay=0.0)
    # model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['accuracy'])

    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    dir = "./image/"
    new_images, im_names = load_image(dir)
    probab_predict = model.predict_proba(new_images, batch_size=1)
    print('probability ', probab_predict)
    result_predict = model.predict_classes(new_images, batch_size=1)
    print('predict ', result_predict, im_names)

    #with open('/home/ubuntu/japan_hw/log/out_predict.txt', 'a') as file:
    #    file.write(str(probab_predict))
    #    file.write(str(result_predict))

    # model.evaluate(X_train, Y_train,batch_size=16)
    # model.fit_generator(datagen.flow(X_train, Y_train, batch_size=16), samples_per_epoch=X_train.shape[0],
    #                     nb_epoch=400, callbacks= callbacks_list, validation_data=(X_test, Y_test))
    # model.save_weights("./model_m61.h5")

def predict(model_name, script, dir = "./image/"):
    json_file = open("./tmp/"+model_name+"_"+script+"_model.json", 'r')
    filepath = "./save/" + model_name + "_" + script + "_weights.best.hdf5"
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(filepath)

    new_images, im_names = load_image(dir)
    probab_predict = model.predict_proba(new_images, batch_size=1)
    print('probability ', probab_predict)
    result_predict = model.predict_classes(new_images, batch_size=1)
    print('predict ', result_predict, im_names)