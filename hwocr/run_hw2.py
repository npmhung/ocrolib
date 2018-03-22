"""Example job for running a neural network."""
import os
import sys
from io import open
import sklearn.metrics as mt
PATH_CUR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(PATH_CUR)
sys.path.append('../../')
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import random
import argparse
import numpy as np
import cv2
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from ocrolib.hwocr import label2code
import json
from scipy import misc
import keras
from keras.models import model_from_json
from keras import optimizers
#from flax.hwocr import  img_prep
from sklearn.model_selection import train_test_split
import pandas

def gen_dict():
    label2codes = pickle.load(open('./data/label2codes.pkl', 'rb'))
    label2chars={}
    chars2label={}
    for l, code in label2codes.items():
        label2chars[l]=label2code.jis2unicode(code)
        chars2label[label2chars[l]]=l
    json.dump(label2chars,open('./data/label2chars.json','w'))
    json.dump(chars2label, open('./data/chars2label.json','w'))


def load_image2(dir, img_rows =64, img_cols = 64):
    im_names = []
    border = 10
    arr_img = np.empty([0,img_rows, img_cols,1])
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for file in sorted(os.listdir(dir)):
        filename = dir + file
        if os.path.isfile(filename) and '.blank' not in file and 'OUTPUT' in file:
            im_names.append(filename)
            img = misc.imread(filename)
            hei, wid, channel = img.shape
            if wid > hei:
                top, bottom = [int(abs(wid - hei) / 2)] * 2
                left, right = 0, 0
            if wid < hei:
                top, bottom = 0, 0
                left, right = [int(abs(wid - hei) / 2)] * 2
            # if(wid <30 and hei< 30):
            top, bottom, left, right = [top + border, bottom + border, left + border, right + border]
            print(top, bottom, left, right)
            color = [255]

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_border = cv2.copyMakeBorder(gray, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            img_border = cv2.resize(img_border, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
            ret, thresh1 = cv2.threshold(img_border, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            size = thresh1.shape
            img_resize = cv2.resize(img_border, size)
            ret, thresh = cv2.threshold(img_resize, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            kernel = np.ones((2, 2), np.uint8)
            img_dilation = cv2.erode(thresh, kernel, iterations=1)

            # rotate = misc.imrotate(img_dilation, -15)
            # img_resize = misc.imresize(invert, size, mode='F')
            # img_resize = cv2.resize(rotate, size)
            out = cv2.normalize(img_dilation.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            reshape = out.reshape([-1, img_rows, img_cols, 1])
            arr_img = np.row_stack((arr_img, reshape))

    arr_img = np.transpose(arr_img, (0, 3, 1, 2))
    print('test shape: ', arr_img.shape)
    return arr_img, im_names

def load_image(dir, img_rows =64, img_cols = 64):
    im_names = []
    arr_img = np.empty([0,img_rows, img_cols,1])
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for file in sorted(os.listdir(dir)):
        filename = dir + file
        if os.path.isfile(filename) and '.blank' not in file and 'OUTPUT' in file:
            im_names.append(filename)
            # print(filename)
            img = cv2.imread(filename, cv2.CV_8UC1)
            # img = cv2.copyMakeBorder(img, top=5, bottom=5, left=5, right=5,
            #                             borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
            # print(img)
            # img = clahe.apply(img)
            out = cv2.resize(img,(img_rows,img_cols),interpolation=cv2.INTER_CUBIC)
            # out=cv2.adaptiveThreshold(out, 1.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
            # # out = cv2.normalize(img_resize, None, 0, 255, cv2.NORM_MINMAX)
            # out = cv2.blur(out, (3,3))
            out = cv2.adaptiveThreshold(out, 1.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
            # out = cv2.medianBlur(out,5)
            # print('shape ', img_resize.shape, ' content', out)
            reshape = out.reshape([-1,img_rows, img_cols,1])
            arr_img = np.row_stack((arr_img, reshape))
    arr_img = np.transpose(arr_img, (0, 3, 1, 2))
    print('test shape: ', arr_img.shape)
    return arr_img, im_names

def load_single_img_old_model(filename, img_rows = 32, img_cols = 32):
    arr_img = np.empty([0, img_rows, img_cols, 1])
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if img is None: return None
    img_resize = cv2.resize(img, (img_cols, img_rows))
    out = cv2.normalize(img_resize.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    reshape = out.reshape([-1, img_rows, img_cols, 1])
    arr_img = np.row_stack((arr_img, reshape))
    return arr_img

import ocrolib.hwocr.mnist_helper as mh

def load_single_img_nice(filename, img_rows =64, img_cols = 64, nobj=3):
    if isinstance(filename, str):
        img = cv2.imread(filename, cv2.CV_8UC1)
    else:
        img=cv2.normalize(filename, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    oldimg=img
    img = mh.do_cropping(img,max_cobj=nobj)
    # img = mh.do_cropping(img)
    # img = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    # img = mh.deskew(img, (img_rows, img_cols))
    img = mh.resize_img(img, (img_rows, img_cols))
    _,img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    # reshape = img.reshape([-1, img_rows, img_cols, 1])
    return oldimg, img.reshape([1, img_rows, img_cols])


def load_single_img(filename, img_rows =64, img_cols = 64):
    # print(filename)
    img = cv2.imread(filename, cv2.CV_8UC1)
    # img = img_prep.deskew(img)
    # img = cv2.copyMakeBorder(img, top=5, bottom=5, left=5, right=5,
    #                             borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
    # print(img)
    # img = clahe.apply(img)
    img2 = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    # out = cv2.medianBlur(img2, 5)
    out = cv2.GaussianBlur(img2, (3, 3), 0)
    #out = img2
    out = cv2.adaptiveThreshold(out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    # out = cv2.Canny(img2, 100, 200)
    # kernel = np.ones((3, 3), np.uint8)
    # out = 255-cv2.erode(255-out, kernel, iterations=1)
    # out[out>0]=1
    out = cv2.normalize(out, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # plt.imshow(out)
    # plt.show()
    # out = cv2.blur(out, (3,3))
    # out = cv2.adaptiveThreshold(out, 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    # im2, contours, hierarchy = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #
    # if len(contours) != 0:
    #     # draw in blue the contours that were founded
    #     cv2.drawContours(img2, contours, -1, 255, 3)
    #
    #     # find the biggest area
    #     c = max(contours, key=cv2.contourArea)
    #
    #     x, y, w, h = cv2.boundingRect(c)
    #     # draw the book contour (in green)
    #     cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    # plt.imshow(out)
    # plt.show()
    # out = cv2.medianBlur(out,5)
    # print('shape ', img_resize.shape, ' content', out)
    reshape = out.reshape([-1, img_rows, img_cols, 1])
    return reshape, out.reshape([1, img_rows, img_cols])

def load_single_img_with_size_info(filename, img_rows =64, img_cols = 64):
    # print(filename)
    img = cv2.imread(filename, cv2.CV_8UC1)
    shape = img.shape
    # img = img_prep.deskew(img)
    # img = cv2.copyMakeBorder(img, top=5, bottom=5, left=5, right=5,
    #                             borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
    # print(img)
    # img = clahe.apply(img)
    img2 = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    # out = cv2.medianBlur(img2, 5)
    out = cv2.GaussianBlur(img2, (3, 3), 0)
    out = cv2.adaptiveThreshold(out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    # out = cv2.Canny(img2, 100, 200)
    # kernel = np.ones((3, 3), np.uint8)
    # out = 255-cv2.erode(255-out, kernel, iterations=1)

    out = cv2.normalize(out, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # plt.imshow(out)
    # plt.show()
    # out = cv2.blur(out, (3,3))
    # out = cv2.adaptiveThreshold(out, 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    # im2, contours, hierarchy = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #
    # if len(contours) != 0:
    #     # draw in blue the contours that were founded
    #     cv2.drawContours(img2, contours, -1, 255, 3)
    #
    #     # find the biggest area
    #     c = max(contours, key=cv2.contourArea)
    #
    #     x, y, w, h = cv2.boundingRect(c)
    #     # draw the book contour (in green)
    #     cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    # plt.imshow(out)
    # plt.show()
    # out = cv2.medianBlur(out,5)
    # print('shape ', img_resize.shape, ' content', out)
    reshape = out.reshape([-1, img_rows, img_cols, 1])
    return reshape, out.reshape([1, img_rows, img_cols]), shape

def load_image_general(dir, img_rows =64, img_cols = 64):
    im_names = []
    arr_img = np.empty([0,img_rows, img_cols,1])
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    print(dir)
    for file in sorted(os.listdir(dir)):
        filename = dir + '/'+file
        if os.path.isfile(filename) and \
                ('.jpg' in filename or '.png' in filename or '.JPG' in filename or '.PNG' in filename):
            print(filename)
            im_names.append(filename)
            reshape, out = load_single_img(filename, img_rows, img_cols)
            arr_img = np.row_stack((arr_img, reshape))
    arr_img = np.transpose(arr_img, (0, 3, 1, 2))
    print('test shape: ', arr_img.shape)
    return arr_img, im_names


def get_extra_data(root_dir='./test_images/form1-20/', dataout='./data/extra_real.pkl',split=0.8):
    all_labels=[]
    all_samples=[]
    print(root_dir)
    mychar2label={}
    mylabel2char={}
    ind=0
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            if '.jpg' in name or '.png' in name or '.JPG' in name or '.PNG' in name:
                filename = os.path.join(root, name)
                print(filename)
                _, img_arrs = load_single_img(filename)
                l = name[-5]
                all_samples+=[img_arrs]
                if l not in mychar2label:
                    mychar2label[l]=ind
                    mylabel2char[ind]=l
                    ind+=1
                all_labels.append(mychar2label[l])


    labels = np.asarray(all_labels)
    labels = keras.utils.to_categorical(labels, len(mychar2label))
    imgs = np.asarray(all_samples)

    print(imgs[:3])

    pickle.dump((imgs[:int(imgs.shape[0]*split)], labels[:int(imgs.shape[0]*split)],
                 imgs[int(imgs.shape[0]*split):], labels[int(imgs.shape[0]*split):]),
                open(dataout, 'wb'))
    json.dump(mylabel2char, open(dataout[:-4]+'.label2chars.json', 'w'))
    json.dump(mychar2label, open(dataout[:-4]+'.chars2label.json', 'w'))
    print(imgs.shape)
    print(labels.shape)
    print(len(mychar2label))
    print(len(mylabel2char))


def load_data_npz(data_dir='./data/fullkata96char.npz', label_dir='./data/katakana.csv', img_rows=64, img_cols=64):
    katamap = open(label_dir, encoding='utf-8')
    label2char={}
    char2label={}
    ind=0
    for l in katamap:
        c=l.split()[0]
        label2char[c]=ind
        char2label[ind]=c
        ind+=1
    print(label2char)
    print(len(char2label))

    height, width = 32, 32
    ary = np.load(data_dir)['a']
    labels = np.load(data_dir)['b']
    print(ary.shape)
    ary = ary.reshape([-1, height, width]).astype(np.float32)
    X=[]
    y=[]
    for i in range(ary.shape[0]):
        im = cv2.normalize(ary[i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        im = cv2.resize(im, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
        # out = cv2.GaussianBlur(im, (3, 3), 0)
        out = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
        kernel = np.ones((3, 3), np.uint8)
        out = 255-cv2.erode(out, kernel, iterations=1)

        out = cv2.normalize(out, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        X.append(out.reshape(1,img_rows,img_cols))
        y.append(labels[i])
        # print(labels[i])
        # plt.imshow(out, cmap='gray')
        # plt.show()



    X_train, X_test, Y_train, Y_test = train_test_split(np.asarray(X), np.asarray(y), test_size=0.2)
    Y_train = keras.utils.to_categorical(Y_train, len(label2char))
    Y_test = keras.utils.to_categorical(Y_test, len(label2char))
    json.dump(char2label, open('./data/full_katakana.char2label.json', 'w'))
    json.dump(label2char, open('./data/full_katakana.label2char.json', 'w'))

    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    pickle.dump((X_train, Y_train, X_test, Y_test), open('./data/full_katakana.pkl', 'wb'))


def get_mnist(out_row=64, out_col=64):
    from keras.datasets import mnist
    from keras import backend as K

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train2=np.zeros((x_train.shape[0], 1, out_row, out_col))
    x_test2=np.zeros((x_test.shape[0], 1, out_row, out_col))

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

    for i in range(x_train.shape[0]):
        img_resize = cv2.resize(x_train[i], (out_row, out_col), interpolation=cv2.INTER_CUBIC)
        # out = cv2.adaptiveThreshold(img_resize, 1.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)
        _, out = cv2.threshold(img_resize, 10, 255, cv2.THRESH_BINARY)
        out = cv2.erode(out, element)
        # out = cv2.medianBlur(out, 3)
        out=cv2.normalize(out, None, 0, 1.0, cv2.NORM_MINMAX)
        out=1-out
        x_train2[i][0]=out
        print(out.shape)
        # plt.imshow(out)
        # plt.show()


    for i in range(x_test.shape[0]):
        img_resize = cv2.resize(x_test[i], (out_row, out_col), interpolation=cv2.INTER_CUBIC)
        # out = cv2.adaptiveThreshold(img_resize, 1.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)
        _, out = cv2.threshold(img_resize, 10, 255, cv2.THRESH_BINARY)
        out = cv2.erode(out, element)
        # out = cv2.medianBlur(out, 3)
        out = cv2.normalize(out, None, 0, 1.0, cv2.NORM_MINMAX)
        out = 1 - out
        x_test2[i][0]=out

    char2label={}
    label2char={}
    for i in range(0,10):
        char2label[chr(i+48)]=i
        label2char[str(i)]=chr(i+48)

    json.dump(char2label, open('./data/full_mnist.char2label.json', 'w'))
    json.dump(label2char, open('./data/full_mnist.label2char.json', 'w'))

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print(x_train2.shape)
    print(x_test2.shape)
    print(y_train.shape)
    print(y_test.shape)
    pickle.dump((x_train2, y_train, x_test2, y_test), open('./data/full_mnist.pkl', 'wb'))


def load_model_weights(name, model):
    try:
        model.load_weights(name)
        print('load ok')
    except Exception as e:
        print ("Can't load weights!")
        print(str(e))


def save_model_weights(name, model):
    try:
        model.save_weights(name)
    except Exception as e:
         print ("failed to save classifier weights")
         print(str(e))

def show_training_images():
    X_train, y_train, X_test, y_test, label2code = pickle.load(open('./data/all.pkl', 'rb'))
    label2imgs = pickle.load(open('./gallery/lib_img.pkl','rb'))
    while 1:
        index = random.choice(range(X_train.shape[0]))
        data = X_train[index]
        id = np.argmax(y_train[index])
        print('shape {} id {} '.format(data.shape, id))

        fig = plt.figure()
        a = fig.add_subplot(1, 2, 1)
        a.set_title('Sample')
        plt.imshow(data[0])
        a = fig.add_subplot(1, 2, 2)
        a.set_title('Gallery')
        plt.imshow(label2imgs[id])
        plt.show()

def make_gallery(fpath='./data/full.pkl'):
    X_train, y_train, X_test, y_test = pickle.load(open(fpath, 'rb'))
    label2imgs={}
    for i in range(X_train.shape[0]):
        if len(y_train.shape)==1:
            id = y_train[i]
        else:
            id = np.argmax(y_train[i])
        if not id in label2imgs:
            label2imgs[id]= X_train[i][0]*255
            # print(np.max(X_train[i][0]))
            # print(np.min(X_train[i][0]))
            cv2.imwrite('./gallery/'+str(id)+'.png', label2imgs[id])
    for i in range(X_test.shape[0]):
        if len(y_test.shape)==1:
            id = y_test[i]
        else:
            id = np.argmax(y_test[i])
        if not id in label2imgs:
            label2imgs[id]= X_test[i][0]*255
            cv2.imwrite('./gallery/'+str(id)+'.png', label2imgs[id])
    pickle.dump(label2imgs,open('./gallery/lib_img.pkl','wb'))
    print('done save gallery!')



def make_gallery10(fpath='./data/full.pkl'):
    X_train, y_train, X_test, y_test = pickle.load(open(fpath, 'rb'))
    print(len(X_train))
    label2imgs={}

    for i in range(X_train.shape[0]-1, -1, -1):
        if len(y_train.shape)==1:
            id = y_train[i]
        else:
            id = np.argmax(y_train[i])
        if not id in label2imgs:
            label2imgs[id]= [X_train[i][0]*255]
            # print(np.max(X_train[i][0]))
            # print(np.min(X_train[i][0]))
        else:
            if len(label2imgs[id])<10:
                label2imgs[id].append(X_train[i][0]*255)

    for i in range(X_train.shape[0]):
        if len(y_train.shape) == 1:
            id = y_train[i]
        else:
            id = np.argmax(y_train[i])
        if not id in label2imgs:
            label2imgs[id] = [X_train[i][0] * 255]
            # print(np.max(X_train[i][0]))
            # print(np.min(X_train[i][0]))
        else:
            if len(label2imgs[id]) < 10:
                label2imgs[id].append(X_train[i][0] * 255)

    # for i in range(X_test.shape[0]-1, -1, -1):
    #     if len(y_test.shape)==1:
    #         id = y_test[i]
    #     else:
    #         id = np.argmax(y_test[i])
    #     if not id in label2imgs:
    #         label2imgs[id]= [X_test[i][0]*255]
    #     else:
    #         if len(label2imgs[id]) < 5:
    #             label2imgs[id].append(X_test[i][0] * 255)

    pickle.dump(label2imgs,open(fpath[:-4]+'.gallery.pkl','wb'))
    print('done save gallery!')


def get_data():
    from .preprocessing.make_keras_input import data
    X_train, y_train, X_test, y_test, label2codes = data(mode='all')
    # print(label2codes)
    pickle.dump((X_train, y_train, X_test, y_test, label2codes), open('./data/all.pkl','wb'))


def predict_single_sep_number(dir_img, model,label2chars, nmodel, nlabel2chars, type=1):
    _, img = load_single_img(dir_img)
    if type ==0:
        print('predict mix')
        probab_predict = model.predict_proba(np.asarray([img]), batch_size=1)
        top_index = np.argsort(probab_predict * -1)[0][0]
        return label2chars[str(top_index)], probab_predict[0][top_index]
    if type==2:
        probab_predict = model.predict_proba(np.asarray([img]), batch_size=1)
        top_index = np.argsort(probab_predict * -1)[0]
        for ind in top_index:
            ch = label2chars[str(ind)]
            if not ch.isdigit() and not ch == '-':
                print('predict char')
                return ch, probab_predict[0][ind]
    if type==1:
        probab_predict = model.predict_proba(np.asarray([img]), batch_size=1)
        top_index = np.argsort(probab_predict * -1)[0]
        t_r = ''
        fid=0
        for ind in top_index:
            ch = label2chars[str(ind)]
            if ch.isdigit() or  ch == '-' or ch =='(' or ch == ')':
                t_r = ch
                fid=ind
                break
        if t_r=='-' or  t_r =='(' or t_r == ')':
            return t_r, probab_predict[0][fid]
        probab_predict = nmodel.predict_proba(np.asarray([img]), batch_size=1)
        print('predict number')
        top_index = np.argsort(probab_predict * -1)[0][0]
        return nlabel2chars[str(top_index)], probab_predict[0][top_index]

def predict_single_sep_number_kata(dir_img,
                                   model,label2chars,
                                   nmodel, nlabel2chars,
                                   kmodel, klabel2chars,
                                   type=0):
    _, img, size = load_single_img_with_size_info(dir_img)
    if type ==0:
        print('predict mix')
        probab_predict = model.predict_proba(np.asarray([img]), batch_size=1)
        top_index = np.argsort(probab_predict * -1)[0][0]
        return label2chars[str(top_index)], probab_predict[0][top_index]

    if type==2:
        print('predict text')
        probab_predict = model.predict_proba(np.asarray([img]), batch_size=1)
        top_index = np.argsort(probab_predict * -1)[0]
        for ind in top_index:
            ch = label2chars[str(ind)]
            if not ch.isdigit() and not ch == '-' and ch not in klabel2chars.values():
                return ch, probab_predict[0][ind]
    if type==1:
        probab_predict = model.predict_proba(np.asarray([img]), batch_size=1)
        top_index = np.argsort(probab_predict * -1)[0]
        t_r = ''
        fid = 0
        for ind in top_index:
            ch = label2chars[str(ind)]
            if ch.isdigit() or  ch == '-' or ch =='(' or ch == ')':
                t_r = ch
                fid = ind
                break
        if t_r=='-' or  t_r =='(' or t_r == ')':
            return t_r, probab_predict[0][fid]
        probab_predict = nmodel.predict_proba(np.asarray([img]), batch_size=1)
        print('predict number')
        top_index = np.argsort(probab_predict * -1)[0][0]
        return nlabel2chars[str(top_index)], probab_predict[0][top_index]

    if type==3:
        # check if image is diacritic
        if max(size) < 50:
            probab_predict = model.predict_proba(np.asarray([img]), batch_size=1)
            top_index = np.argsort(probab_predict * -1)[0][:2]
            t_r = ''
            fid = 0
            for ind in top_index:
                ch = label2chars[str(ind)]
                print(ch)
                if ch in ['\'', '0', 'o']:
                    t_r = ch
                    fid = ind
                    break
            if t_r == '\'':
                return u'\u3099', probab_predict[0][fid]
            elif t_r in ['0', 'o']:
                return u'\u309A', probab_predict[0][fid]
        print('predict kata')
        probab_predict = kmodel.predict_proba(np.asarray([img]), batch_size=1)
        top_index = np.argsort(probab_predict * -1)[0][0]
        return klabel2chars[str(top_index)], probab_predict[0][top_index]

def predict_single_sep_number_kata_kanji(dir_img,
                                   model,label2chars,
                                   nmodel, nlabel2chars,
                                   kmodel, klabel2chars,
                                   khmodel, khlabel2chars,
                                   type=0):
    if type != 4:
        _, img, size = load_single_img_with_size_info(dir_img)
    else:
        _, img = load_single_img(dir_img)
        #img = load_single_img_old_model(dir_img)
    if type ==0:
        print('predict mix')
        probab_predict = model.predict_proba(np.asarray([img]), batch_size=1)
        top_index = np.argsort(probab_predict * -1)[0][0]
        return label2chars[str(top_index)], probab_predict[0][top_index]

    if type==2:
        print('predict text')
        probab_predict = model.predict_proba(np.asarray([img]), batch_size=1)
        top_index = np.argsort(probab_predict * -1)[0]
        for ind in top_index:
            ch = label2chars[str(ind)]
            if not ch.isdigit() and not ch == '-' and ch not in klabel2chars.values():
                return ch, probab_predict[0][ind]
    if type==1:
        probab_predict = model.predict_proba(np.asarray([img]), batch_size=1)
        top_index = np.argsort(probab_predict * -1)[0]
        t_r = ''
        fid = 0
        for ind in top_index:
            ch = label2chars[str(ind)]
            if ch.isdigit() or  ch == '-' or ch =='(' or ch == ')':
                t_r = ch
                fid = ind
                break
        if t_r=='-' or  t_r =='(' or t_r == ')':
            return t_r, probab_predict[0][fid]
        probab_predict = nmodel.predict_proba(np.asarray([img]), batch_size=1)
        print('predict number')
        top_index = np.argsort(probab_predict * -1)[0][0]
        return nlabel2chars[str(top_index)], probab_predict[0][top_index]

    if type==3:
        # check if image is diacritic
        if max(size) < 50:
            probab_predict = model.predict_proba(np.asarray([img]), batch_size=1)
            top_index = np.argsort(probab_predict * -1)[0][:2]
            t_r = ''
            fid = 0
            for ind in top_index:
                ch = label2chars[str(ind)]
                print(ch)
                if ch in ['\'', '0', 'o']:
                    t_r = ch
                    fid = ind
                    break
            if t_r == '\'':
                return u'\u3099', probab_predict[0][fid]
            elif t_r in ['0', 'o']:
                return u'\u309A', probab_predict[0][fid]
        print('predict kata')
        probab_predict = kmodel.predict_proba(np.asarray([img]), batch_size=1)
        top_index = np.argsort(probab_predict * -1)[0][0]
        return klabel2chars[str(top_index)], probab_predict[0][top_index]

    # predict kanji
    if type==4:
        probab_predict = khmodel.predict_proba(np.asarray([img]), batch_size=1)
        print('predict kanji')
        top_index = np.argsort(probab_predict * -1)[0][0]
        return khlabel2chars[str(top_index)], probab_predict[0][top_index]
        #return label2code.label2unicode_etl9(top_index), probab_predict[0][top_index]

def predict_single_sep_number_kata_kanji_old(dir_img,
                                   model,label2chars,
                                   nmodel, nlabel2chars,
                                   kmodel, klabel2chars,
                                   khmodel, khlabel2chars,
                                   type=0):
    if type != 4:
        _, img, size = load_single_img_with_size_info(dir_img)
    else:
        img = load_single_img_old_model(dir_img)
    if type ==0:
        print('predict mix...')
        probab_predict = model.predict_proba(np.asarray([img]), batch_size=1)
        top_index = np.argsort(probab_predict * -1)[0][0]
        return top_index, label2chars[str(top_index)], probab_predict[0][top_index]

    if type==2:
        print('predict text...')
        probab_predict = model.predict_proba(np.asarray([img]), batch_size=1)
        top_index = np.argsort(probab_predict * -1)[0]
        for ind in top_index:
            ch = label2chars[str(ind)]
            if not ch.isdigit() and not ch == '-' and ch not in klabel2chars.values():
                return ind, ch, probab_predict[0][ind]
    if type==1:
        probab_predict = model.predict_proba(np.asarray([img]), batch_size=1)
        top_index = np.argsort(probab_predict * -1)[0]
        t_r = ''
        fid = 0
        for ind in top_index:
            ch = label2chars[str(ind)]
            if ch.isdigit() or  ch == '-' or ch =='(' or ch == ')':
                t_r = ch
                fid = ind
                break
        if t_r=='-' or  t_r =='(' or t_r == ')':
            return fid, t_r, probab_predict[0][fid]
        probab_predict = nmodel.predict_proba(np.asarray([img]), batch_size=1)
        print('predict number...')
        top_index = np.argsort(probab_predict * -1)[0][0]
        return top_index, nlabel2chars[str(top_index)], probab_predict[0][top_index]

    if type==3:
        # check if image is diacritic
        if max(size) < 50:
            probab_predict = model.predict_proba(np.asarray([img]), batch_size=1)
            top_index = np.argsort(probab_predict * -1)[0][:2]
            t_r = ''
            fid = 0
            for ind in top_index:
                ch = label2chars[str(ind)]
                #print(ch)
                if ch in ['\'', '0', 'o']:
                    t_r = ch
                    fid = ind
                    break
            if t_r == '\'':
                return fid, u'\u3099', probab_predict[0][fid]
            elif t_r in ['0', 'o']:
                return fid, u'\u309A', probab_predict[0][fid]
        print('predict kata...')
        probab_predict = kmodel.predict_proba(np.asarray([img]), batch_size=1)
        top_index = np.argsort(probab_predict * -1)[0][0]
        return top_index, klabel2chars[str(top_index)], probab_predict[0][top_index]

    # predict kanji
    if type==4:
        probab_predict = khmodel.predict_proba(img, batch_size=1)
        print('predict kanji')
        top_index = np.argsort(probab_predict * -1)[0][0]
        return label2code.label2unicode_etl9(top_index), probab_predict[0][top_index]

def predict_single(dir_img,model, label2chars, type=0):
    _, img = load_single_img(dir_img)
    print(img.shape)
    # plt.imshow(img[0])
    # plt.show()
    if type==0:
        result_predict = model.predict_classes(np.asarray([img]), batch_size=1)
    else:
        probab_predict = model.predict_proba(np.asarray([img]), batch_size=1)
        top_index = np.argsort(probab_predict * -1)[0]
        for ind in top_index:
            ch = label2chars[str(ind)]
            if ch.isdigit() or ch=='-':
                if type==1:#digit
                    print('predict digit')
                    return ch
            else:
                if type==2:#char
                    print('predict char')
                    return ch
    return label2chars[str(result_predict[0])]

def predict_single_topk(dir_img,model, topk=3):
    _, img = load_single_img(dir_img)
    print(img.shape)
    probab_predict = model.predict_proba(np.asarray([img]), batch_size=1)
    top_index = np.argsort(probab_predict * -1)[0][:topk]
    return top_index, img

def predict(dir_img, topk=3):
    from keras.models import model_from_json
    label2imgs = pickle.load(open('./data/full.gallery.pkl', 'rb'))
    label2chars = json.load(open('./data/full.label2char.json'))
    # print(label2codes)
    json_file = open("./save/model7_1.json", 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    load_model_weights('./save/M7_1-all_weights.h5', model)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['accuracy'])
    new_images, im_names = load_image_general(dir_img)
    f = open('./test.txt','w',encoding='utf-8')
    f.write('...\n')
    for i in range(new_images.shape[0]):
        bimg=np.asarray(np.asarray([new_images[i]]))
        print(im_names[i])
        probab_predict = model.predict_proba(bimg, batch_size=1)
        top_index = np.argsort(probab_predict * -1)[0][:topk]
        result_predict = top_index[0]
        # print('probability ', probab_predict)
        # result_predict = model.predict_classes(bimg, batch_size=1)
        jc=label2chars[str(result_predict)]
        f.write(str(jc))
        f.write('\n')
        print('predict {} -->{} with prob {}'.format(result_predict, jc.encode('utf-8'),
                                                     probab_predict[0][result_predict]))
        fig = plt.figure()
        raw = cv2.imread(im_names[i])
        a = fig.add_subplot(topk + 1, 5, 1)
        a.set_title('Real input')
        plt.imshow(raw)
        a = fig.add_subplot(topk + 1, 5, 2)
        a.set_title('Processed input')
        plt.imshow(new_images[i][0])
        curindj=6
        for jj in range(5):
            a = fig.add_subplot(topk + 1, 5,curindj)
            a.set_title('Top 1')
            plt.imshow(label2imgs[result_predict][jj])
            curindj+=1

        print('probability ', probab_predict)

        for ii, index in enumerate(list(top_index[1:])):
            jc =label2chars[str(index)]
            print('predict {} -->{} '.format(index, jc.encode('utf-8')))
            for jj in range(5):
                a = fig.add_subplot(topk + 1, 5, curindj)
                a.set_title('Top {}'.format(ii+2))
                plt.imshow(label2imgs[index][jj])
                curindj+=1
        fig.tight_layout()
        plt.show()
    f.close()

def report(dir_input, dir_report, topk=3):
    if not os.path.isdir(dir_report):
        os.mkdir(dir_report)
    from keras.models import model_from_json
    label2chars = json.load(open('./data/label2chars.json'))
    label2imgs = pickle.load(open('./gallery/lib_img.pkl', 'rb'))
    json_file = open("./save/model7_1.json", 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    load_model_weights('./save/M7_1-hiragana_weights.h5', model)

    all_acc=0
    all_sam=0
    for subdir in sorted(os.listdir(dir_input)):
        subrdir = dir_report + '/' + subdir
        subdir = dir_input + '/' + subdir

        if not os.path.isdir(subrdir):
            os.mkdir(subrdir)
        if os.path.isdir(subdir):
            for subdir1 in sorted(os.listdir(subdir)):
                subrdir1 = subrdir + '/' + subdir1 + '/'
                subdir1 = subdir + '/' + subdir1+'/'



                if os.path.isdir(subdir1):
                    acc_folder = 0
                    print(subdir1)
                    new_images, im_names = load_image_general(subdir1)
                    for i in range(new_images.shape[0]):
                        bimg = np.asarray([new_images[i]])
                        print(im_names[i])

                        real_jc = os.path.basename(im_names[i])[:-4].split('-')[-1]
                        res_str = real_jc
                        # result_predict = model.predict_classes(bimg, batch_size=1)[0]
                        probab_predict = model.predict_proba(bimg, batch_size=1)
                        top_index = np.argsort(probab_predict*-1)[0][:topk]
                        result_predict = top_index[0]
                        jc = label2chars[str(result_predict)]
                        print('predict {} -->{} '.format(result_predict, jc.encode('utf-8')))
                        res_str+=' vs '+jc
                        if real_jc==jc:
                            print('correct')
                            acc_folder+=1
                        else:
                            print('wrong')
                        fig = plt.figure()
                        a = fig.add_subplot(1, topk+1, 1)
                        a.set_title('Real input')
                        plt.imshow(new_images[i][0])
                        a = fig.add_subplot(1, topk+1, 2)
                        a.set_title('Predict Class')
                        plt.imshow(label2imgs[result_predict])


                        print('probability ', probab_predict)


                        for ii, index in enumerate(list(top_index[1:])):
                            jc = label2chars[str(index)]
                            print('predict {} -->{} '.format(index, jc.encode('utf-8')))
                            res_str += ' vs ' + jc
                            a = fig.add_subplot(1, topk+1, ii+3)
                            a.set_title('Suggest Class')
                            plt.imshow(label2imgs[index])
                        plt.show()

                        if not os.path.isdir(subrdir1):
                            os.mkdir(subrdir1)

                        if real_jc==label2chars[str(result_predict)]:
                            res_str+=' --> correct'
                        else:
                            res_str += ' --> wrong'

                        fig.savefig(subrdir1+'/reuslt{}.jpg'.format(os.path.basename(im_names[i])[:-4]))
                        with open(subrdir1+'/reuslt{}.txt'.format(format(os.path.basename(im_names[i])[:-4])),'w', encoding='utf-8') as f:
                            f.write(res_str)
                    with open(subrdir1 + '/reuslt.txt', 'w', encoding='utf-8') as f:
                        f.write(str(acc_folder/new_images.shape[0]))
                        f.write('\r\n')
                        f.write(str(acc_folder)+' vs '+str(new_images.shape[0]))
                    all_acc+=acc_folder
                    all_sam+=new_images.shape[0]
    with open(dir_report + '/result.txt', 'w', encoding='utf-8') as f:
        f.write(str(all_acc/all_sam))
        f.write('\r\n')
        f.write(str(all_acc) + ' vs ' + str(all_sam))
    print(all_acc/all_sam)


def combine_data2(data1, data2, dataout, char2label1, label2char2):
    char2label1 = json.load(open(char2label1))
    label2char2 = json.load(open(label2char2))
    X_train, y_train, X_test, y_test = pickle.load(open(data1, 'rb'))
    print(y_train.shape)
    print(X_train.shape)
    print(X_train[0])
    print(np.max(X_train[0]))
    print(np.min(X_train[0]))
    X_train2, y_train2, X_test2, y_test2 = pickle.load(open(data2, 'rb'))
    print(y_train2.shape)
    print(X_train2.shape)
    X_train_all = np.concatenate((X_train, X_train2), axis=0)
    X_test_all = np.concatenate((X_test, X_test2), axis=0)
    print(label2char2)
    y_train22=np.zeros(y_train2.shape[0], dtype=np.int32)
    new_labels=len(char2label1)
    for yind in range(y_train2.shape[0]):
        ly = y_train2[yind]
        if len(y_train2.shape)>1:
            ly = np.argmax(y_train2[yind])
        if label2char2[str(ly)] in char2label1:
            y_train22[yind]= char2label1[label2char2[str(ly)]]
            # print(label2char2[str(ly)])
            # print(char2label1[label2char2[str(ly)]])
        else:
            print('new label')
            y_train22[yind] =new_labels
            char2label1[label2char2[str(ly)]] = new_labels
            new_labels+=1

    y_test22 = np.zeros(y_test2.shape[0], dtype=np.int32)
    for yind in range(y_test2.shape[0]):
        ly = y_test2[yind]
        if len(y_test2.shape) > 1:
            ly = np.argmax(y_test2[yind])
        if label2char2[str(ly)] in char2label1:
            y_test22[yind]= char2label1[label2char2[str(ly)]]
        else:
            y_test22[yind] =new_labels
            char2label1[label2char2[str(ly)]] = new_labels
            new_labels+=1

    nlabel2chars={}
    for k, v in char2label1.items():
        nlabel2chars[v]=k

    if len(y_train.shape)>1:
        y_train_all = np.concatenate((np.argmax(y_train, axis=1), y_train22), axis=0)
    else:
        y_train_all = np.concatenate((y_train, y_train22), axis=0)

    if len(y_test.shape) > 1:
        y_test_all = np.concatenate((np.argmax(y_test,axis=1), y_test22), axis=0)
    else:
        y_test_all = np.concatenate((y_test, y_test22), axis=0)

    print(y_train_all.shape)
    print(y_test_all.shape)
    # while True:
    #     ind = random.choice(range(X_train_all.shape[0]))
    #     print(nlabel2chars[y_train_all[ind]])
    #     plt.imshow(X_train_all[ind][0])
    #     plt.show()
    pickle.dump((X_train_all, y_train_all, X_test_all, y_test_all), open(dataout, 'wb'),protocol=4)
    json.dump(char2label1,open(dataout[:-4]+'.char2label.json','w'))
    json.dump(nlabel2chars, open(dataout[:-4] + '.label2char.json', 'w'))
    print(len(char2label1))
    print(len(nlabel2chars))
    make_gallery(dataout)

def combine_data3(data1, data2, dataout, cury=['kanhi','kata'], sub=1.0):

    X_train, y_train, X_test, y_test = pickle.load(open(data1, 'rb'))
    sublen=int(X_train.shape[0]*sub)
    X_train=X_train[:sublen]
    X_test=X_test[:(sublen//5)]
    y_train = y_train[:sublen]
    y_test = y_test[:(sublen//5)]
    print(y_train.shape)
    print(X_train.shape)
    print(X_train[0])
    X_train2, y_train2, X_test2, y_test2 = pickle.load(open(data2, 'rb'))
    print(y_train2.shape)
    print(X_train2.shape)
    X_train_all = np.concatenate((X_train, X_train2), axis=0)
    X_test_all = np.concatenate((X_test, X_test2), axis=0)

    y_train_all = np.zeros((X_train_all.shape[0]), dtype=np.int32)

    if len(cury)>2:
        for i in range(y_train.shape[0]):
            y_train_all[i]=y_train[i]

    for i in range(y_train.shape[0],y_train_all.shape[0]):
        y_train_all[i]=len(cury)-1

    y_test_all = np.zeros((X_test_all.shape[0]), dtype=np.int32)
    if len(cury) > 2:
        for i in range(y_test.shape[0]):
            y_test_all[i] = y_test[i]
    for i in range(y_test.shape[0], y_test_all.shape[0]):
        y_test_all[i] = len(cury)-1

    char2label1={}
    nlabel2chars={}
    c=0
    for yn in cury:
        char2label1[yn]=c
        nlabel2chars[c]=yn
        c+=1



    print(y_train_all.shape)
    print(y_test_all.shape)
    # while True:
    #     ind = random.choice(range(X_train_all.shape[0]))
    #     print(nlabel2chars[y_train_all[ind]])
    #     plt.imshow(X_train_all[ind][0])
    #     plt.show()
    print('start_dump...')
    pickle.dump((X_train_all, y_train_all, X_test_all, y_test_all), open(dataout, 'wb'),protocol=4)
    make_gallery10(dataout)
    json.dump(char2label1,open(dataout[:-4]+'.char2label.json','w'))
    json.dump(nlabel2chars, open(dataout[:-4] + '.label2char.json', 'w'))
    print(len(char2label1))
    print(len(nlabel2chars))


def combine_data():
    X_train, y_train, X_test, y_test, label2codes = pickle.load(open('./data/all.pkl', 'rb'))
    print(y_train.shape)
    X_train2, y_train2, X_test2, y_test2 = pickle.load(open('./data/mnist.pkl', 'rb'))
    print(y_train2.shape)
    # for x in X_train2:
    #     plt.imshow(x[0])
    #     plt.show()
    #     plt.imshow(X_train[0][0])
    #     plt.show()
    X_train_all = np.concatenate((X_train, X_train2), axis=0)
    X_test_all = np.concatenate((X_test, X_test2), axis=0)



    for i in range(10):
        label2codes[y_train.shape[1]+i]=48+i


    y_train_all = np.zeros((y_train.shape[0] + y_train2.shape[0], y_train.shape[1] + y_train2.shape[1]))
    print(y_train_all.shape)
    y_train_all[:y_train.shape[0],:y_train.shape[1]] = y_train
    y_train_all[y_train.shape[0]:,y_train.shape[1]:] = y_train2

    y_test_all = np.zeros((y_test.shape[0] + y_test2.shape[0], y_test.shape[1] + y_test2.shape[1]))
    print(y_test_all.shape)
    y_test_all[:y_test.shape[0],:y_test.shape[1]] = y_test
    y_test_all[y_test.shape[0]:,y_test.shape[1]:] = y_test2

    print(X_train_all.shape)
    print(y_train_all.shape)
    print(X_test_all.shape)
    print(y_test_all.shape)

    pickle.dump((X_train_all, y_train_all, X_test_all, y_test_all), open('./data/combine.pkl', 'wb'),protocol=4)
    pickle.dump(label2codes,open('./data/label2codes.pkl', 'wb'))
    make_gallery('./data/combine.pkl')


def remake_data_nice(data_dir='./data/mnist_full.pkl', nobj=3):
    print('---START REMAKE DATA {} with {} objs-------'.format(data_dir, nobj))
    X_train, y_train, X_test, y_test = pickle.load(open(data_dir, 'rb'))
    m=os.path.basename(data_dir)[:-4]
    indexs=list(range(X_train.shape[0]))
    # random.shuffle(indexs)
    for i in indexs:
        oldimg,img=load_single_img_nice(X_train[i][0], nobj=nobj)
        X_train[i][0]=img
        # plt.subplot(1, 2, 1)
        # plt.imshow(img[0])
        # plt.subplot(1, 2, 2)
        # plt.imshow(oldimg)
        # plt.show()

    indexs = list(range(X_test.shape[0]))
    for i in indexs:
        oldimg,img=load_single_img_nice(X_test[i][0], nobj=nobj)
        X_test[i][0]=img
    print('---START DUMP NEW DATA-----')
    pickle.dump((X_train, y_train, X_test, y_test), open('./data/{}.nice.pkl'.format(m), 'wb'), protocol=4)
    print('done!')


def remake_data_noise(data_dir, nobj=5, noise_ratio=0.2):
    print('---START REMAKE DATA {} with {} objs-------'.format(data_dir, nobj))
    import noise_gen
    bg_ims = noise_gen.collect_noise_bg()
    print('done load noise {}'.format(len(bg_ims)))
    X_train, y_train, X_test, y_test = pickle.load(open(data_dir, 'rb'))
    m = os.path.basename(data_dir)[:-4]
    indexs = list(range(X_train.shape[0]))
    # random.shuffle(indexs)
    nnoise=0
    for i in indexs:
        if np.random.rand()<noise_ratio:
            nnoise+=1
            oldimg = X_train[i][0]
            ran_noise = np.random.choice(bg_ims, 1)[0]
            # print(ran_noise)
            rh = random.randint(-20, 20)
            rv = random.randint(-20, 20)
            img=noise_gen.add_blend_img(ran_noise, oldimg, shift_hor=0, shift_ver=0)
            # print(img.shape)
            _,img=load_single_img_nice(img, nobj=nobj)
            # print(img.shape)
            X_train[i][0] = img
            # plt.subplot(1, 2, 1)
            # plt.imshow(img[0])
            # plt.subplot(1, 2, 2)
            # plt.imshow(oldimg)
            # plt.show()
    print('nnoise {} vs total {}'.format(nnoise,len(X_train)))

    indexs = list(range(X_test.shape[0]))
    for i in indexs:
        if np.random.rand()<noise_ratio:
            oldimg = X_test[i][0]
            ran_noise = np.random.choice(bg_ims, 1)[0]
            # print(ran_noise)
            img = noise_gen.add_blend_img(ran_noise, oldimg)
            # print(img.shape)
            _, img = load_single_img_nice(img, nobj=nobj)
            X_test[i][0] = img
    print('---START DUMP NEW DATA-----')
    pickle.dump((X_train, y_train, X_test, y_test), open('./data/{}.noise.pkl'.format(m), 'wb'), protocol=4)
    print('done!')


def check_data(data_dir, num=1000, store_path='./test_images/check/'):
    label2char = json.load(open(data_dir[:-4] + '.label2char.json'))
    X_train, y_train, X_test, y_test = pickle.load(open(data_dir, 'rb'))
    # if len(y_train.shape)==1:
    #     y_train = keras.utils.to_categorical(y_train, len(char2label))
    #     y_test= keras.utils.to_categorical(y_test, len(char2label))
    m = os.path.basename(data_dir)[:-4]
    store=store_path+'/'+m+'/'
    if not os.path.isdir(store):
        os.mkdir(store)

    indexs = list(range(X_train.shape[0]))
    random.shuffle(indexs)
    for ci,i in enumerate(indexs):
        im=X_train[i][0]
        l = label2char[str(y_train[i])]
        # print(np.max(im))
        # print(np.min(im))
        cv2.imwrite(store+'/{}{}.png'.format(l,ci),im*255)
        if ci>num:
            break

    indexs = list(range(X_test.shape[0]))
    for ci, i in enumerate(indexs):
        im = X_test[i][0]
        l = label2char[str(y_test[i])]
        cv2.imwrite(store + '/{}{}.png'.format(l, num+ci), im*255)
        if ci > num:
            break

def train(train_dir='./data/full.pkl', save_m ="./save/model7_1.json", save_w='./save/M7_1-all_weights.h5'):
    from keras.callbacks import ModelCheckpoint
    from ocrolib.hwocr.models import M7_1
    from keras import optimizers
    char2label = json.load(open(train_dir[:-4]+'.char2label.json'))
    X_train, y_train, X_test, y_test = pickle.load(open(train_dir,'rb'))

    if len(y_train.shape)==1:
        y_train = keras.utils.to_categorical(y_train, len(char2label))
        y_test= keras.utils.to_categorical(y_test, len(char2label))

    n_output = len(char2label)
    model = M7_1(n_output=n_output, input_shape=(1, 64, 64))

    load_model_weights(save_w, model)
    checkpoint = ModelCheckpoint(save_w, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    adam = optimizers.Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    model_json = model.to_json()
    with open(save_m, "w") as json_file:
        json_file.write(model_json)
    model.fit(X_train, y_train, shuffle=True,
              epochs=100, callbacks= [checkpoint],
              batch_size=2048, validation_data=(X_test[:X_test.shape[0]//2], y_test[:y_test.shape[0]//2]))

    score, acc = model.evaluate(X_test[X_test.shape[0]//2:], y_test[y_test.shape[0]//2:],
                                batch_size=512,
                                verbose=1)
    print ("Training size: ", X_train.shape[0])
    print ("Test size: ", X_test.shape[0])
    print ("Test Score: ", score)
    print ("Test Accuracy: ", acc)

def test(train_dir='./data/full.pkl', save_m ="./save/model7_1.json", save_w='./save/M7_1-all_weights.h5',
         store_path='./test_images/test_model_predict/'):
    label2char = json.load(open(train_dir[:-4] + '.label2char.json'))
    print(len(label2char))
    X_train, y_train, X_test, y_test = pickle.load(open(train_dir, 'rb'))
    if len(y_test.shape)>1:
        y_test=np.argmax(y_test,axis=-1)
    m = os.path.basename(train_dir)[:-4]
    store = store_path + '/' + m + '/'
    if not os.path.isdir(store):
        os.mkdir(store)


    json_file = open(save_m, 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

    load_model_weights(save_w, model)
    adam = optimizers.Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    indexs = list(range(X_test.shape[0]))
    random.shuffle(indexs)

    c=0
    acc=0
    for i in indexs:
        img=X_test[i]
        result_predict = model.predict_classes(np.asarray([img]), batch_size=1)[0]
        result_str = label2char[str(result_predict)]
        real_str=label2char[str(y_test[i])]
        str_file='{} vs {}-{}'.format(result_str,real_str,c)
        if real_str==result_str:
            acc+=1
        plt.imshow(img[0])
        plt.show()
        if c<500:
            cv2.imwrite(store + '/{}.png'.format(str_file), img[0] * 255)
        else:
            break
        c+=1
    print('acc {}'.format(acc/c))

def eval_save_img(train_dir='./data/full.pkl', save_m ="./save/model7_1.json", save_w='./save/M7_1-all_weights.h5'):
    label2imgs = pickle.load(open(train_dir[:-4]+'.gallery.pkl', 'rb'))
    label2chars = json.load(open(train_dir[:-4] + '.label2char.json'))
    X_train, y_train, X_test, y_test = pickle.load(open(train_dir, 'rb'))
    # X_test=X_test[:100]
    if len(y_test.shape) == 1:
        # y_train = keras.utils.to_categorical(y_train, len(label2chars))
        y_test = keras.utils.to_categorical(y_test, len(label2chars))
    json_file = open(save_m, 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

    load_model_weights(save_w, model)
    adam = optimizers.Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    # score, acc = model.evaluate(X_test[X_test.shape[0] // 2:], y_test[y_test.shape[0] // 2:],
    #                         batch_size=512,
    #                         verbose=1)
    # print(acc)
    topk=3
    folder_out='./test_images/{}_wrong/'.format(os.path.basename(train_dir)[:-4])
    list_wrong={}
    num_wrong=0
    probab_predict2 = model.predict_proba(X_test, batch_size=512, verbose=True)
    top_indexs = np.argsort(probab_predict2 * -1,axis=-1)[:,:topk]
    labels = np.argmax(y_test, axis=-1)
    for i in range(X_test.shape[0]):
        if i%1000==0:
            print('num wrong {} vs done {}/{}'.format(num_wrong,i,X_test.shape[0]))

        top_index=top_indexs[i]

        result_predict = top_index[0]
        # print('probability ', probab_predict)
        # result_predict = model.predict_classes(bimg, batch_size=1)
        jc=label2chars[str(result_predict)]
        rlabel=labels[i]
        # print('predict {} -->{} vs {}'.format(result_predict, jc.encode('utf-8'), rlabel))
        if rlabel!=result_predict:
            # print('probability ', probab_predict)
            if num_wrong>1000:
                break
            if rlabel not in list_wrong:
                list_wrong[rlabel]=[]
            if len(list_wrong[rlabel])<3:
                num_wrong += 1
                list_wrong[rlabel].append(X_test[i][0])
                fig = plt.figure()
                a = fig.add_subplot(1, topk + 1, 1)
                a.set_title('Real({})'.format(rlabel))
                plt.imshow(X_test[i][0], cmap='gray')
                a = fig.add_subplot(1, topk + 1, 2)
                a.set_title('Predict({})'.format(result_predict))
                plt.imshow(label2imgs[result_predict], cmap='gray')

                # print('probability ', probab_predict)

                for ii, index in enumerate(list(top_index[1:])):
                    jc =label2chars[str(index)]
                    # print('predict {} -->{} '.format(index, jc.encode('utf-8')))
                    a = fig.add_subplot(1, topk + 1, ii + 3)
                    a.set_title('Suggest({})'.format(index))
                    plt.imshow(label2imgs[index], cmap='gray')

                plt.show()
                dir_path=folder_out+'/'+str(rlabel)+'/'
                if not os.path.isdir(folder_out):
                    os.mkdir(folder_out)
                if not os.path.isdir(dir_path):
                    os.mkdir(dir_path)
                fig.tight_layout()
                fig.savefig(dir_path+ '/{}.report.jpg'.format(len(list_wrong[rlabel])))




def eval(train_dir='./data/full.pkl', save_m ="./save/model7_1.json", save_w='./save/M7_1-all_weights.h5'):
    # label2imgs = pickle.load(open(train_dir[:-4]+'.gallery.pkl', 'rb'))
    from ocrolib.hwocr.models import M7_1
    label2chars = json.load(open(train_dir[:-4] + '.label2char.json'))
    X_train, y_train, X_test, y_test = pickle.load(open(train_dir, 'rb'))
    if len(y_test.shape) == 1:
        # y_train = keras.utils.to_categorical(y_train, len(label2chars))
        y_test = keras.utils.to_categorical(y_test, len(label2chars))
    json_file = open(save_m, 'r')
    n_output = len(label2chars)
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model = M7_1(n_output=n_output, input_shape=(1, 64, 64))
    load_model_weights(save_w, model)
    adam = optimizers.Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    # score, acc = model.evaluate(X_test[X_test.shape[0] // 2:], y_test[y_test.shape[0] // 2:],
    #                         batch_size=512,
    #                         verbose=1)
    # print(acc)
    topk=3
    folder_out='./test_images/kanhi_test_wrong/'
    list_wrong={}
    df=pandas.DataFrame(columns=('Real', 'Predict', 'Suggest1', 'Suggest2'))
    loci=0
    num_wrong=0
    probab_predict2 = model.predict_proba(X_test, batch_size=512, verbose=True)
    top_indexs = np.argsort(probab_predict2 * -1,axis=-1)[:,:topk]
    labels = np.argmax(y_test, axis=-1)
    for i in range(X_test.shape[0]):
        if i%1000==0:
            print('num wrong {} vs done {}/{}'.format(num_wrong,i,X_test.shape[0]))

        top_index=top_indexs[i]

        result_predict = top_index[0]
        # print('probability ', probab_predict)
        # result_predict = model.predict_classes(bimg, batch_size=1)
        jc=label2chars[str(result_predict)]
        rlabel=labels[i]
        # print('predict {} -->{} vs {}'.format(result_predict, jc.encode('utf-8'), rlabel))
        if rlabel!=result_predict:
            val = [rlabel]
            # print('probability ', probab_predict)
            num_wrong+=1
            for ii, index in enumerate(list(top_index)):
                val+=[index]
            df.loc[loci]=val
            loci+=1
                # if len(list_wrong[rlabel])<3:
            #     list_wrong[rlabel].append(X_test[i][0])
            #     fig = plt.figure()
            #     a = fig.add_subplot(1, topk + 1, 1)
            #     a.set_title('Real({})'.format(rlabel))
            #     plt.imshow(X_test[i][0], cmap='gray')
            #     a = fig.add_subplot(1, topk + 1, 2)
            #     a.set_title('Predict({})'.format(result_predict))
            #     plt.imshow(label2imgs[result_predict][0], cmap='gray')
            #
            #     # print('probability ', probab_predict)
            #
            #     for ii, index in enumerate(list(top_index[1:])):
            #         jc =label2chars[str(index)]
            #         # print('predict {} -->{} '.format(index, jc.encode('utf-8')))
            #         a = fig.add_subplot(1, topk + 1, ii + 3)
            #         a.set_title('Suggest({})'.format(index))
            #         plt.imshow(label2imgs[index][0], cmap='gray')
            #
            #     plt.show()
            #     dir_path=folder_out+'/'+str(rlabel)+'/'
            #     if not os.path.isdir(dir_path):
            #         os.mkdir(dir_path)
            #     fig.tight_layout()
            #     fig.savefig(dir_path+ '/{}.report.jpg'.format(len(list_wrong[rlabel])))
    print('err {}'.format(num_wrong/X_test.shape[0]))
    df.to_csv('./test_images/{}.csv'.format(os.path.basename(train_dir[:-4])))

def trim_test_data(train_dir='./data/full.pkl'):
    import ocrolib.hwocr.erasestroke.erasestroke as es
    X_train, y_train, X_test, y_test = pickle.load(open(train_dir, 'rb'))
    X_test_trim=[]
    indexs = list(range(X_test.shape[0]))
    label2imgs = {}
    for i in indexs:
        if i%1000==0:
            print('done {}/{}'.format(i, len(indexs)))
        # plt.imshow(X_test[i][0])
        # plt.show()
        try:
            newx=es.erosedStroke(X_test[i][0])
            _, newx=load_single_img_nice(newx)
            X_test_trim.append(newx)
            if len(y_test.shape) == 1:
                id = y_test[i]
            else:
                id = np.argmax(y_test[i])
            if not id in label2imgs:
                label2imgs[id] = newx[0] * 255
        except:
            print('st wrong')
        # print(newx.shape)
        # plt.imshow(newx[0])
        # plt.show()
    X_test_trim = np.asarray(X_test_trim)
    print(X_test_trim.shape)
    pickle.dump((None, None, X_test_trim, y_test), open(train_dir[:-4]+'.trim.pkl', 'wb'), protocol=4)
    pickle.dump(label2imgs, open(train_dir[:-4] + '.trim.gallery.pkl', 'wb'))
    print('done save gallery!')


def report2(root_dir, label2chard = './data/full.label2char.json',
                  save_m ="./save/model7_1.json", save_w='./save/M7_1-all_weights.h5'):
    print(root_dir)
    from keras.models import model_from_json
    print('start load...')
    label2chars = json.load(open(label2chard))
    json_file = open(save_m, 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    load_model_weights(save_w,  model)
    count=0
    wrong=0
    list_predict=[]
    list_real=[]
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            folder = root.split(os.sep)[-2]
            print(folder)
            # if '.predict' in name:
            #     if 'x' in name:
            #         wrong+=1
            if 'DAVID' not in name and 'line' not in name and 'checkbox' not in name:
                filename = os.path.join(root, name)
                print(filename)
                if filename[-4:]!='.txt' and \
                        ('.jpg' in filename or '.png' in filename or '.JPG' in filename or '.PNG' in filename):
                    oldimg, img = load_single_img_nice(filename)

                    count+=1
                    result_predict = model.predict_classes(np.asarray([img]), batch_size=1)[0]
                    result_str = label2chars[str(result_predict)]
                    print('{} -->{}'.format(result_predict,result_str))
                    correct_label=os.path.basename(root.split(os.sep)[-2])
                    # print(root.split(os.sep))
                    print(correct_label)
                    if correct_label!=result_str:
                        wrong+=1
                    list_predict.append(result_str)
                    list_real.append(correct_label)
                    plt.subplot(1,2,1)
                    plt.imshow(img[0])
                    plt.subplot(1, 2, 2)
                    plt.imshow(oldimg)
                    plt.show()
                    with open(filename[:-4]+'.txt','w', encoding='utf-8') as f:
                        f.write(result_str)
                    # with open(filename[:-4]+'.{}.predict'.format(result_str),'w', encoding='utf-8') as f:
                    #     f.write(result_str)
    print(mt.classification_report(list_real, list_predict))
    print('wrong {}/{} -->{}'.format(wrong,count,wrong/count))

def report2nissay(root_dir, label2chard = './data/full.label2char.json',
                  save_m ="./save/model7_1.json", save_w='./save/M7_1-all_weights.h5'):
    print(root_dir)
    from keras.models import model_from_json
    label2chars = json.load(open(label2chard))
    json_file = open(save_m, 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    load_model_weights(save_w,  model)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['accuracy'])
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            type=0
            if 'DAVID' not in name and 'line' not in name and 'checkbox' not in name :
                if 'number' in root.split(os.path.sep)[-1]:
                    type=1
                if 'text' in  root.split(os.path.sep)[-1]:
                    type=2
                filename = os.path.join(root, name)
                print(filename)
                if filename[-4:]!='.txt' and \
                        ('.jpg' in filename or '.png' in filename or '.JPG' in filename or '.PNG' in filename):
                    str_re=predict_single(filename, model, label2chars, type)
                    with open(filename[:-4]+'.txt','w', encoding='utf-8') as f:
                        f.write(str_re)


def report2nissay_number_sep(root_dir, data_dir):
    label2chard = data_dir + '/data/full.label2char.json'
    save_m = data_dir + "/data/model_hw_7_1.json"
    save_w = data_dir + '/weight/weight_7_1.h5'

    nlabel2chard = data_dir + '/data/full_mnist.label2char.json'
    nsave_m = data_dir + "/data/model_mnist.json"
    nsave_w = data_dir + '/weight/weight_mnist.h5'

    #report2nissay_number_sep_origin(root_dir, )

def report_extract_excel(root_dir, label2chard = './data/mnist_full.nice.noise.label2char.json',
                  save_m ="./save/model_mnist_full.nice.noise.json",
                         save_w='./save/weight_mnist_full.nice.noise.h5',
                         out_wrong='./test_images/excel_wrong/', use_vae=False):
    if use_vae:
        import vae as nvae
        print('...use nvae...')
        vae, encoder, generator = nvae.build_denoise_model_cnn()

        load_model_weights('./save/vae/weight_kata.noise.h5', vae)
        # print(vae.summary())
    print(root_dir)
    from keras.models import model_from_json
    print('start load...')
    label2chars = json.load(open(label2chard))
    json_file = open(save_m, 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    load_model_weights(save_w,  model)
    count=c=0
    acc=0
    list_predict=[]
    list_real=[]
    out_wrong=out_wrong+'/'+save_m.split(os.sep)[-1][:-4]
    if not os.path.isdir(out_wrong):
        os.mkdir(out_wrong)
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            # folder = root.split(os.sep)[-2]
            # print(folder)
            # if '.predict' in name:
            #     if 'x' in name:
            #         wrong+=1
            if '.png' in name.lower():
                filename = os.path.join(root, name)
                print(filename)
                _, img = load_single_img_nice(filename, nobj=5)
                real_str = name[:-4].split('-')[-1]
                if len(real_str) > 1:
                    print('double label??? {}'.format(real_str))
                    continue
                if use_vae:
                    # print(img.shape)
                    old_img=img
                    img = generator.predict(encoder.predict(np.asarray([img.reshape(img.shape[1], img.shape[2],1)]), batch_size=1),
                                                       verbose=1) \
                        .reshape(1, 1, img.shape[1], img.shape[2])[0]
                    ret, im = cv2.threshold(img[0]*255, 150, 255, cv2.THRESH_BINARY)
                    img=[im/255]
                    # _, img=load_single_img_nice(im, nobj=5)
                result_predict = model.predict_classes(np.asarray([img]), batch_size=1)[0]
                result_str = label2chars[str(result_predict)]
                if real_str.isspace() or not real_str:
                    real_str="'"
                list_predict.append(result_str)
                list_real.append(real_str)
                print('{} vs {}'.format(result_str, real_str))
                if len(result_str)>1 and real_str in result_str:
                        acc+=1
                elif result_str==real_str:
                    acc+=1
                else:
                    str_file = '{} vs {}-{}'.format(result_str, real_str, c)
                    if c < 1500:
                        if use_vae:
                            fig = plt.figure()
                            a = fig.add_subplot(1, 2, 1)
                            a.set_title('Noise')
                            plt.imshow(old_img[0] *255, cmap='gray')
                            a = fig.add_subplot(1, 2, 2)
                            a.set_title('Denoise')
                            plt.imshow(img[0] * 255, cmap='gray')
                            fig.savefig(out_wrong + '/{}.png'.format(str_file))
                            plt.close()
                        else:
                            cv2.imwrite(out_wrong + '/{}.png'.format(str_file), img[0] * 255)
                    c+=1
                count+=1
    print('acc {}, {} vs {}'.format(acc/count, acc, count))



def load_model_report2nissay_number_sep(data_dir):
    # init data folder
    # label2code.init_char_map(data_dir)

    label2chard = data_dir + '/data/full.label2char.json'
    save_m = data_dir + "/data/model_hw_7_1.json"
    save_w = data_dir + '/weight/weight_7_1.h5'

    nlabel2chard = data_dir + '/data/full_mnist.label2char.json'
    nsave_m = data_dir + "/data/model_mnist.json"
    nsave_w = data_dir + '/weight/weight_mnist.h5'

    print("Loading OCR model... ")

    from keras.models import model_from_json
    label2chars = json.load(open(label2chard))
    json_file = open(save_m, 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    load_model_weights(save_w, model)

    nlabel2chars = json.load(open(nlabel2chard))
    json_file = open(nsave_m, 'r')
    nloaded_model_json = json_file.read()
    nmodel = model_from_json(nloaded_model_json)
    load_model_weights(nsave_w, nmodel)

    print("Done.")
    return (model, nmodel, label2chars, nlabel2chars)

def load_model_report2nissay_number_sep_kata(data_dir):
    # init data folder
    label2chard = data_dir + '/data/full.label2char.json'
    save_m = data_dir + "/data/model_hw_7_1.json"
    save_w = data_dir + '/weight/weight_7_1.h5'

    nlabel2chard = data_dir + '/data/full_mnist.label2char.json'
    nsave_m = data_dir + "/data/model_mnist.json"
    nsave_w = data_dir + '/weight/weight_mnist.h5'

    klabel2chard = data_dir + '/data/full_katakana.label2char.json'
    ksave_m = data_dir + "/data/model_katakana.json"
    ksave_w = data_dir + '/weight/weight_katakana.h5'

    print("Loading OCR model... ")

    from keras.models import model_from_json
    label2chars = json.load(open(label2chard))
    json_file = open(save_m, 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    load_model_weights(save_w, model)

    nlabel2chars = json.load(open(nlabel2chard))
    json_file = open(nsave_m, 'r')
    nloaded_model_json = json_file.read()
    nmodel = model_from_json(nloaded_model_json)
    load_model_weights(nsave_w, nmodel)

    klabel2chars = json.load(open(klabel2chard))
    json_file = open(ksave_m, 'r')
    kloaded_model_json = json_file.read()
    kmodel = model_from_json(kloaded_model_json)
    load_model_weights(ksave_w, kmodel)

    print("Done.")
    return (model, nmodel, kmodel, label2chars, nlabel2chars, klabel2chars)

def load_model_report2nissay_number_sep_kata_kanji(data_dir):

    # init data folder
    label2chard = data_dir + '/data/full_nissay.label2char.json'
    save_m = data_dir + "/save/model_nissay.json"
    save_w = data_dir + '/save/weight_nissay.h5'

    nlabel2chard = data_dir + '/data/full_mnist.label2char.json'
    nsave_m = data_dir + "/save/model_mnist.json"
    nsave_w = data_dir + '/save/weight_mnist.h5'

    klabel2chard = data_dir + '/data/full_katakana.label2char.json'
    ksave_m = data_dir + "/save/model_katakana.json"
    ksave_w = data_dir + '/save/weight_katakana.h5'

    khlabel2chard = data_dir + "/data/kanji_hira.label2char.json"
    khsave_m = data_dir + "/save/model_kanji_hira.json"
    khsave_w = data_dir + "/save/weight_kanji_hira.h5"

    print("Loading OCR model... ")

    from keras.models import model_from_json
    label2chars = json.load(open(label2chard))
    json_file = open(save_m, 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    load_model_weights(save_w, model)

    nlabel2chars = json.load(open(nlabel2chard))
    json_file = open(nsave_m, 'r')
    nloaded_model_json = json_file.read()
    nmodel = model_from_json(nloaded_model_json)
    load_model_weights(nsave_w, nmodel)

    klabel2chars = json.load(open(klabel2chard))
    json_file = open(ksave_m, 'r')
    kloaded_model_json = json_file.read()
    kmodel = model_from_json(kloaded_model_json)
    load_model_weights(ksave_w, kmodel)

    khlabel2chars = json.load(open(khlabel2chard))
    json_file = open(khsave_m, 'r')
    khloaded_model_json = json_file.read()
    khmodel = model_from_json(khloaded_model_json)
    load_model_weights(khsave_w, khmodel)

    # test old model
    # label2code.init_char_map_kanji(data_dir)
    # from flax.hwocr.models import M8
    # khlabel2chars = json.load(open(data_dir + "/test_images/tadashi_model/data/kanji_hira.label2char.json"))
    # khsave_w = data_dir + "/test_images/tadashi_model/save/weights_m8_etl9.h5"
    # khmodel = M8(n_output=3036, input_shape=(32,32,1))
    # khmodel.summary()
    # khmodel.load_weights(khsave_w)

    print("Done.")
    return model, nmodel, kmodel, khmodel, label2chars, nlabel2chars, klabel2chars, khlabel2chars


def report2nissay_number_kata_sep(root_dir_list, model_data):
    model, nmodel, kmodel, label2chars, nlabel2chars, klabel2chars = model_data
    for root_dir in root_dir_list:
        for root, dirs, files in os.walk(root_dir):
            for name in files:
                type = 0
                if 'DAVID' not in name and 'line' not in name and 'checkbox' not in name:
                    if 'number' in root.split(os.path.sep)[-1]:
                        type = 1
                    if 'text' in root.split(os.path.sep)[-1]:
                        type = 2
                    if 'kata' in root.split(os.path.sep)[-1]:
                        type=3
                    filename = os.path.join(root, name)
                    print(filename)
                    if filename[-4:] != '.txt' and \
                            ('.jpg' in filename or '.png' in filename or '.JPG' in filename or '.PNG' in filename):
                        str_re, prob = predict_single_sep_number_kata(filename,
                                                                      model, label2chars,
                                                                      nmodel, nlabel2chars,
                                                                      kmodel, klabel2chars,
                                                                      type)
                        with open(filename[:-4] + '.txt', 'w', encoding='utf-8') as f:
                            f.write(str_re)
                        with open(filename[:-4] + '_probability.txt', 'wt') as f:
                            try:
                                f.write(str(prob))
                            except:
                                f.write(unicode(prob))

def report2nissay_number_sep_kata_kanji_origin(root_dir_list, model_data):
    model, nmodel, kmodel, khmodel, label2chars, nlabel2chars, klabel2chars, khlabel2chars = model_data
    for root_dir in root_dir_list:
        for root, dirs, files in os.walk(root_dir):
            for name in files:
                type = 0
                if 'DAVID' not in name and 'line' not in name and 'checkbox' not in name:
                    if 'number' in root.split(os.path.sep)[-1]:
                        type = 1
                    if 'text' in root.split(os.path.sep)[-1]:
                        type = 2
                    if 'kata' in root.split(os.path.sep)[-1]:
                        type = 3
                    if 'kanji' in root.split(os.path.sep)[-1]:
                        type = 4
                    filename = os.path.join(root, name)
                    print(filename)
                    if filename[-4:] != '.txt' and \
                            ('.jpg' in filename or '.png' in filename or '.JPG' in filename or '.PNG' in filename):
                        ind, str_re, prob = predict_single_sep_number_kata_kanji_old(filename,
                                                                      model,  label2chars,
                                                                      nmodel, nlabel2chars,
                                                                      kmodel, klabel2chars,
                                                                      khmodel, khlabel2chars,
                                                                      type)
                        with open(filename[:-4] + '.txt', 'w', encoding='utf-8') as f:
                            f.write(str_re)
                        with open(filename[:-4] + '_probability.txt', 'wt') as f:
                            try:
                                f.write(str(prob))
                            except:
                                f.write(unicode(prob))

def report2nissay_number_sep_kata_kanji_accuracy_test(root_dir, data_dir):
    import csv
    import configparser
    import os
    import unicodedata

    config = configparser.ConfigParser()

    model, nmodel, kmodel, khmodel, label2chars, nlabel2chars, klabel2chars, khlabel2chars = load_model_report2nissay_number_sep_kata_kanji(data_dir)
    type_name = {1:'number', 2:'text', 3:'kata', 4:'kanji'}
    results = []

    for root, dirs, files in os.walk(root_dir):
        for name in files:
            type = 0
            if 'line' not in name and 'checkbox' not in name and 'mixedline' not in name and 'labeled' in root.split(os.path.sep)[-1]:
                template_index = root.split(os.path.sep)[-5][-1]
                field_index = root.split(os.path.sep)[-3]
                line_index = root.split(os.path.sep)[-2][-1]

                # read config file to determine field type
                path_config = '%s/meta/temp%s/' % (data_dir, template_index)
                config_filename = '%sconfig%s.ini' % (path_config, field_index)
                if os.path.isfile(config_filename):
                    config.read(config_filename)
                    isLineNumber = config.getint('textnumber', 'line%snumber' % line_index)
                else:
                    isLineNumber = -1

                if 'number' in root.split(os.path.sep)[-2]:
                    type = 1
                if 'text' in root.split(os.path.sep)[-2]:
                    type = 2
                if 'kata' in root.split(os.path.sep)[-2] or isLineNumber == 2:
                    type = 3
                if 'kanji' in root.split(os.path.sep)[-2] or isLineNumber == 3:
                    type = 4
                filename = os.path.join(root, name)
                print(filename)
                if filename[-4:] != '.txt' and \
                        ('.jpg' in filename or '.png' in filename or '.JPG' in filename or '.PNG' in filename):
                    ind, str_re, prob = predict_single_sep_number_kata_kanji_old(filename,
                                                                            model, label2chars,
                                                                            nmodel, nlabel2chars,
                                                                            kmodel, klabel2chars,
                                                                            khmodel, khlabel2chars,
                                                                            type)
                    str_re = unicodedata.normalize('NFKC', str_re)
                    true_label = filename[-5]
                    if true_label != 'X':
                        results.append({'filename':filename, 'template': root.split(os.path.sep)[-5], 'ocr_label':ind,
                                       'type':type_name[type], 'true_label':true_label, 'ocr_result':str_re, 'prob':prob})
    # export as csv
    #results = pickle.load(open('results.pkl','rb'))
    #print(results)
    pickle.dump(results, open('results.pkl','wb'))
    keys = results[0].keys()
    with open('accuracy_test.csv', 'w', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, keys, delimiter=',')
        dict_writer.writeheader()
        dict_writer.writerows(results)
    print('Done')

def extract_wrong_case(dir_save='./test_images/wrong_cases_img'):
    from shutil import copyfile
    import scipy.misc


    nlabel2imgs = pickle.load(open('./data/mnist_full.gallery.pkl', 'rb'))
    nchar2label = json.load(open('./data/mnist.char2label.json'))
    klabel2imgs = pickle.load(open('./data/full_katakana_quote.gallery.pkl', 'rb'))
    kchar2label = json.load(open('./data/full_katakana_quote.char2label.json'))
    khlabel2imgs = pickle.load(open('./data/kanji_hira.gallery.pkl', 'rb'))
    khchar2label = json.load(open('./data/kanji_hira.char2label.json'))
    label2img={"text":khlabel2imgs,"number":nlabel2imgs,"kata":klabel2imgs}
    char2label = {"text": khchar2label, "number": nchar2label, "kata": kchar2label}
    results=pickle.load(open('results.pkl','rb'))
    acc=0
    for i,r in enumerate(results):
        if r['true_label']=='-':
            continue
        prefix='true'
        if r['true_label']!=r['ocr_result']:
            prefix='false'
        else:
            acc+=1
        fname = dir_save+'/'+prefix
        if not os.path.isdir(fname):
            os.mkdir(fname)
        fname = fname + '/' + r['type']
        if not os.path.isdir(fname):
            os.mkdir(fname)
        cfo = fname + '/' + r['true_label']
        cfn = cfo + '/' + os.path.basename(r['filename'])
        if not os.path.isdir(cfo):
            os.mkdir(cfo)
        copyfile(r['filename'], cfn)

        # print(r['true_label'].encode('unicode-escape'))

        try:
            l = int(char2label['number'][r['true_label']])
            lim = nlabel2imgs
        except:
            try:
                l = int(char2label['kata'][r['true_label']])
                lim = klabel2imgs
            except:
                try:
                    lim = label2img['text']
                    l = int(char2label['text'][r['true_label']])
                except:
                    print('st wrong {} {}'.format(r['true_label'], r['true_label'].encode('unicode-escape')))
                    # continue
        for ii in range(5):
            cfntr = cfn + '.train{}'.format(ii) + cfn[-4:]
            # print(label2img[r['type']][l])
            # print(lim.keys())
            # print(type(l))
            scipy.misc.imsave(cfntr, lim[l][ii])
    print(acc/len(results))
    print(acc)
    print(len(results))


def report2nissay_number_sep_origin2(root_dir_list, label2chard = './data/full.label2char.json',
                  save_m ="./save/model7_1.json", save_w='./save/M7_1-all_weights.h5',
                             nlabel2chard='./data/full_mnist.label2char.json',
                             nsave_m="./save/model_mnist.json", nsave_w='./save/weight_mnist.h5'):


    print("Loading OCR model... ")

    from keras.models import model_from_json
    label2chars = json.load(open(label2chard))
    json_file = open(save_m, 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    load_model_weights(save_w, model)

    nlabel2chars = json.load(open(nlabel2chard))
    json_file = open(nsave_m, 'r')
    nloaded_model_json = json_file.read()
    nmodel = model_from_json(nloaded_model_json)
    load_model_weights(nsave_w, nmodel)

    print(root_dir_list)
    for root_dir in root_dir_list:
        for root, dirs, files in os.walk(root_dir):
            for name in files:
                type=0
                if 'DAVID' not in name and 'line' not in name and 'checkbox' not in name:
                    if 'number' in root.split(os.path.sep)[-1]:
                        type=1
                    if 'text' in  root.split(os.path.sep)[-1]:
                        type=2
                    # if 'mix' in root.split(os.path.sep)[-1]:
                    #     type=0
                    filename = os.path.join(root, name)
                    print(filename)
                    if filename[-4:]!='.txt' and \
                            ('.jpg' in filename or '.png' in filename or '.JPG' in filename or '.PNG' in filename):
                        str_re, prob =predict_single_sep_number(filename, model, label2chars, nmodel, nlabel2chars, type)
                        with open(filename[:-4]+'.txt','w', encoding='utf-8') as f:
                            f.write(str_re)
                        with open(filename[:-4] + '_probability.txt', 'wt') as f:
                            try:
                                f.write(str(prob))
                            except:
                                f.write(unicode(prob))

def report2nissay_number_sep_origin(root_dir_list, model_data):
    model, nmodel, label2chars, nlabel2chars = model_data
    print(root_dir_list)
    for root_dir in root_dir_list:
        for root, dirs, files in os.walk(root_dir):
            for name in files:
                type=0
                if 'DAVID' not in name and 'line' not in name and 'checkbox' not in name:
                    if 'number' in root.split(os.path.sep)[-1]:
                        type=1
                    if 'text' in  root.split(os.path.sep)[-1]:
                        type=2
                    filename = os.path.join(root, name)
                    print(filename)
                    if filename[-4:]!='.txt' and \
                            ('.jpg' in filename or '.png' in filename or '.JPG' in filename or '.PNG' in filename):
                        str_re, prob =predict_single_sep_number(filename, model, label2chars, nmodel, nlabel2chars, type)
                        with open(filename[:-4]+'.txt','w', encoding='utf-8') as f:
                            f.write(str_re)
                        with open(filename[:-4] + '_probability.txt', 'wt') as f:
                            f.write(str(prob))

def report_images(root_dir):
    print(root_dir)
    from keras.models import model_from_json
    label2imgs = pickle.load(open('./data/full.gallery.pkl', 'rb'))
    json_file = open("./save/save/model7_1.json", 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    load_model_weights('./save/save/M7_1-hiragana_weights.h5', model)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['accuracy'])
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            if 'DAVID' not in name:
                filename = os.path.join(root, name)
                print(filename)
                if filename[-4:] != '.txt' and \
                        ('.jpg' in filename or '.png' in filename or '.JPG' in filename or '.PNG' in filename):
                    topk=3
                    top_index, imgp = predict_single_topk(filename, model, topk)
                    result_predict = top_index[0]
                    fig = plt.figure()
                    raw = cv2.imread(filename)
                    a = fig.add_subplot(topk + 1, 5, 1)
                    a.set_title('Real input')
                    plt.imshow(raw)
                    a = fig.add_subplot(topk + 1, 5, 2)
                    a.set_title('Processed input')
                    plt.imshow(imgp[0])
                    curindj = 6
                    random.shuffle(label2imgs[result_predict])
                    for jj in range(5):
                        a = fig.add_subplot(topk + 1, 5, curindj)
                        a.set_title('Top 1')
                        plt.imshow(label2imgs[result_predict][jj])
                        curindj += 1
                    for ii, index in enumerate(list(top_index[1:])):
                        random.shuffle(label2imgs[index])
                        for jj in range(5):
                            a = fig.add_subplot(topk + 1, 5, curindj)
                            a.set_title('Top {}'.format(ii + 2))
                            plt.imshow(label2imgs[index][jj])
                            curindj += 1
                    fig.tight_layout()
                    # plt.show()
                    fig.savefig(filename[:-4]+'.report.jpg')


def split_kanji_hira():
    X_train, y_train, X_test, y_test = pickle.load(open('./data/kanji_hira.nice.pkl', 'rb'))
    mylabel2char = json.load(open('./data/kanji_hira.label2char.json'))
    kanji_label2char={}
    kanji_char2label={}
    X_traink, y_traink, X_testk, y_testk=[],[],[],[]


    hira_label2char={}
    hira_char2label={}
    X_trainh, y_trainh, X_testh, y_testh = [],[],[],[]


    indexs = list(range(X_train.shape[0]))
    chira=0
    ckanji=0
    for i in indexs:
        label = y_train[i]
        if label<=70:#hira
            if mylabel2char[str(label)] not in hira_char2label:
                hira_char2label[mylabel2char[str(label)]]=chira
                hira_label2char[str(chira)]=mylabel2char[str(label)]
                chira += 1

            X_trainh.append(X_train[i])
            y_trainh.append(hira_char2label[mylabel2char[str(label)]])

        else:
            if mylabel2char[str(label)] not in kanji_char2label:
                kanji_char2label[mylabel2char[str(label)]] = ckanji
                kanji_label2char[str(ckanji)] = mylabel2char[str(label)]
                ckanji += 1

            X_traink.append(X_train[i])
            y_traink.append(kanji_char2label[mylabel2char[str(label)]])

    indexs = list(range(X_test.shape[0]))
    for i in indexs:
        label = y_test[i]
        if label <= 70:  # hira
            if mylabel2char[str(label)] not in hira_char2label:
                hira_char2label[mylabel2char[str(label)]] = chira
                hira_label2char[str(chira)] = mylabel2char[str(label)]
                chira += 1

            X_testh.append(X_test[i])
            y_testh.append(hira_char2label[mylabel2char[str(label)]])

        else:
            if mylabel2char[str(label)] not in kanji_char2label:
                kanji_char2label[mylabel2char[str(label)]] = ckanji
                kanji_label2char[str(ckanji)] = mylabel2char[str(label)]
                ckanji += 1

            X_testk.append(X_test[i])
            y_testk.append(kanji_char2label[mylabel2char[str(label)]])

        # print('{} --> {}'.format(str(label), mylabel2char[str(label)]))
        # plt.imshow(X_train[i][0])
        # plt.show()

    X_trainh=np.asarray(X_trainh)
    X_testh = np.asarray(X_testh)
    y_trainh = np.asarray(y_trainh)
    y_testh = np.asarray(y_testh)

    print(X_trainh.shape)
    print(X_testh.shape)
    print(y_trainh.shape)
    print(y_testh.shape)
    print(len(hira_label2char))

    X_traink = np.asarray(X_traink)
    X_testk = np.asarray(X_testk)
    y_traink = np.asarray(y_traink)
    y_testk = np.asarray(y_testk)

    print(X_traink.shape)
    print(X_testk.shape)
    print(y_traink.shape)
    print(y_testk.shape)
    print(len(kanji_label2char))


    print('start_dump hira...')
    dataout='./data/hiragana.nice.pkl'
    pickle.dump((X_trainh, y_trainh, X_testh, y_testh), open( dataout,'wb'),protocol=4)
    json.dump(hira_char2label,open(dataout[:-4]+'.char2label.json','w'))
    json.dump(hira_label2char, open(dataout[:-4] + '.label2char.json', 'w'))

    print('start_dump kanji...')
    dataout = './data/kanji.nice.pkl'
    pickle.dump((X_traink, y_traink, X_testk, y_testk), open(dataout, 'wb'), protocol=4)
    json.dump(kanji_char2label, open(dataout[:-4] + '.char2label.json', 'w'))
    json.dump(kanji_label2char, open(dataout[:-4] + '.label2char.json', 'w'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="show", help="train/predict/report/show/get_data/make_gallery/make_mnist/combine/gen_dict")
    args = parser.parse_args()
    if args.mode=='train':
        print('train mode...')
        train('./data/full.pkl')
    if args.mode=='show':
        print('show mode')
        show_training_images()
    if args.mode=='get_data':
        print('get_data mode...')
        get_data()
    if args.mode=='predict':
        print('predict mode...')
        predict('./test_images/Yamato_input/form3/word/')
    if args.mode=='make_gallery':
        print('make_gallery mode...')
        make_gallery()
    if args.mode=='make_mnist':
        print('maket_mnis mode...')
        get_mnist()
    if args.mode=='combine':
        print('combine mode...')
        combine_data()
    if args.mode=='report':
        print('report mode...')
        report('./test_images/form1-20', './report/form1-20')
    if args.mode=='gen_dict':
        print('gen_dict mode...')
        gen_dict()

if __name__=='__main__':
    # main()
    # train('./data/extra_img.pkl')
    #get_extra_data()
    # make_gallery()
    # get_mnist()
    # combine_data()
    #combine_data2('./data/combine.pkl','./data/extra_real.pkl','./data/full.pkl',
    #                './data/chars2label.json','./data/extra_real.label2chars.json')
    #combine_data2('./data/full.pkl','./seed_new_labels/quote.pkl','./data/full_nissay.pkl',
    #            './data/full.char2label.json','./seed_new_labels/quote.label2char.json')

    # combine_data2('./data/mnist_full.nice.pkl','./seed_new_labels/only_hyp.pkl','./data/mnist_hyp.nice.pkl',
    #             './data/full_mnist.char2label.json','./seed_new_labels/only_hyp.label2char.json')

    # combine_data2('./data/mnist_hyp.nice.pkl','./seed_new_labels/strange_number.pkl','./data/mnist_hyp_strange.nice.pkl',
    #             './data/mnist_hyp.nice.char2label.json','./seed_new_labels/strange_number.label2char.json')

    # combine_data3('./data/kanji_hira.nice.pkl','./data/full_katakana_quote.nice.pkl','./data/kh_kata.pkl',
    #        cury=['kanhi','kata'], sub=0.2)
    #
    #
    #
    # combine_data3('./data/kh_kata.pkl', './data/mnist_full.nice.pkl', './data/kh_kata_mnist.pkl',
    #                cury=['kanhi','kata','mnist'])


    # combine_data2('./data/hiragana.nice.pkl','./data/full_katakana_quote.nice.pkl','./data/hira_kata.pkl',
    #            './data/hiragana.nice.char2label.json','./data/full_katakana_quote.nice.label2char.json')
    #
    # combine_data3('./data/kanji.nice.pkl', './data/hira_kata.pkl', './data/kanji_hika.pkl',
    #        cury=['kanji','hika'], sub=0.2)
    #
    # combine_data3('./data/kanji_hika.pkl', './data/mnist_full.nice.pkl', './data/kanji_hika_mnist.pkl',
    #                cury=['kanji','hika','mnist'])

    # make_gallery10('./data/mnist_full.nice.pkl')

    # train('./data/full_nissay.pkl',"./save/model_nissay.json","./save/weight_nissay.h5")
    # train('./data/full_mnist.pkl', "./save/model_mnist.json", "./save/weight_mnist.h5")
    # train('./data/kh_kata_mnist.pkl', "./save/model_kh_kata_mnist.json", "./save/weight_kh_kata_mnist.h5")
    # train('./data/kanji_hika_mnist.pkl', "./save/model_kanji_hika_mnist.json", "./save/weight_kanji_hika_mnist.h5")
    #train('./data/mnist_hyp.nice.pkl', "./save/model_mnist_hyp.nice.json", "./save/weight_mnist_hyp.nice.h5")

    # train('./data/kanji.nice.pkl', "./save/model_kanji.nice.json", "./save/weight_kanji.nice.h5")

    # train('./data/hira_kata.pkl', "./save/model_hira_kata.json", "./save/weight_hira_kata.h5")

    # test('./data/mnist_full.nice.noise.pkl', "./save/model_mnist_full.nice.noise.json",
    #      "./save/weight_mnist_full.nice.noise.h5")
    # eval('./data/kanji_hira.trim.pkl', "./save/model_kanji_hira.json", "./save/weight_kanji_hira.h5")

    # eval_save_img('./data/kanji_hira.trim.pkl', "./save/model_kanji_hira.json", "./save/weight_kanji_hira.h5")

    # eval()
    # predict('./test_images/yamato_form1-20.2/word1/line1')
    # report2('./test_images/yamato_form1-20.2/')
    # remake_data_nice('./data/mnist_full.pkl')
    # remake_data_nice('./data/full_katakana_quote.pkl', 5)
    # remake_data_nice('./data/kanji_hira.pkl',5)

    # remake_data_noise('./data/kanji.nice.pkl', noise_ratio=0.1)

    check_data('./data/kanji_hira.pkl')
    # split_kanji_hira()

    # trim_test_data('./data/kanji_hira.pkl')

    # report2('./test_images/lt_test_data/',
    #         './data/kanji_hika_mnist.label2char.json',
    #         './save/model_kanji_hika_mnist.json',
    #         './save/weight_kanji_hika_mnist.h5')

    # report2('./test_images/lt_test_data/kanji/',
    #         './data/kanji.nice.label2char.json',
    #         './save/model_kanji.nice.json',
    #         './save/weight_kanji.nice.h5')


    # report2('./test_images/dnp_cut_images_171108/',
    #         './data/mnist_hyp.nice.label2char.json',
    #         './save/model_mnist_hyp.nice.json',
    #         './save/weight_mnist_hyp.nice.h5')
    # report2nissay('./test_images/runhocr/',
    #               './data/full_nissay.label2char.json',
    #               './save/model_nissay.json',
    #               './save/weight_nissay.h5')
    # report2nissay_number_sep_origin2(['./test_images/sougou20019/'],
    #               './data/full_nissay.label2char.json',
    #               './save/model_nissay.json',
    #               './save/weight_nissay.h5',
    #               './data/full_mnist.label2char.json',
    #               './save/model_mnist.json',
    #               './save/weight_mnist.h5')
    # report2nissay_number_sep_kata_kanji_accuracy_test('./test_images/Template03','./')
    # extract_wrong_case()
    # make_gallery10(fpath='./data/mnist_full.pkl')

    # report_images('./test_images/Pros_Yamato_Data_Input.img/',)

    # report_extract_excel('./test_images/excel_ims_number/',label2chard = './data/mnist_hyp.nice.label2char.json',
    #               save_m ="./save/model_mnist_hyp.nice.json", save_w='./save/weight_mnist_hyp.nice.h5')

    # report_extract_excel('./test_images/excel_ims_kata/', label2chard='./data/full_katakana_quote.nice.label2char.json',
    #                      save_m="./save/model_katakana_quote.nice.json",
    #                      save_w='./save/weight_katakana_quote.nice.h5', use_vae=True)

    # report_extract_excel('./test_images/excel_ims_kanji/', label2chard='./data/kanji.nice.label2char.json',
    #                      save_m="./save/model_kanji.nice.json", save_w='./save/weight_kanji.nice.h5')

    # X_train, y_train, X_test, y_test = pickle.load(open('./data/kanji_hika_mnist.pkl', 'rb'))
    # mylabel2char = json.load(open('./data/kanji_hika_mnist.label2char.json'))
    # print(len(mylabel2char))
    # indexs = list(range(X_train.shape[0]))
    # # labels = np.argsort(y_train,-1)
    # # indexs = list(labels)
    # random.shuffle(indexs)
    # # unique_labels=[]
    # for i in indexs:
    #     label=y_train[i]
    #     # if label not in unique_labels:
    #     # unique_labels.append(label)
    #     print('{} --> {}'.format(str(label),mylabel2char[str(label)]))
    #     plt.imshow(X_train[i][0])
    #     plt.show()
    # indexs = list(range(X_test.shape[0]))
    # for i in indexs:
    #     label=y_train[i]


    # load_data_npz()
    # train('./data/full_katakana.pkl', "./save/model_katakana.json", "./save/weight_katakana.h5")
    # train('./data/full_katakana_quote.nice.pkl', "./save/model_katakana_quote.nice.json", "./save/weight_katakana_quote.nice.h5")
    # train('./data/kanji_hira.pkl', "./save/model_kanji_hira.json", "./save/weight_kanji_hira.h5")
    # train('./data/mnist_full.nice.noise.pkl', "./save/model_mnist_full.nice.noise.json", "./save/weight_mnist_full.nice.noise.h5")
    # train('./data/mnist_hyp_strange.nice.pkl', "./save/model_mnist_hyp_strange.nice.json", "./save/weight_mnist_hyp_strange.nice.h5")
