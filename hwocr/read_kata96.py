import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import json
import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl
# mpl.use('Agg')

def preproces_data(im):
    out = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    kernel = np.ones((3, 3), np.uint8)
    out = 255 - cv2.erode(out, kernel, iterations=1)

    out = cv2.normalize(out, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return out

def load_data_npz(data_dir='./data/fullkata96char.npz', label_dir='./data/katakana.csv', img_rows=64, img_cols=64):
    katamap = open(label_dir, encoding='utf-8')
    label2char={}
    char2label={}
    ind=0
    for l in katamap:
        c=l.split()[0]
        char2label[c]=ind
        label2char[ind]=c
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

        out = preproces_data(im)

        X.append(out.reshape(1,img_rows,img_cols))
        y.append(labels[i])
        # print(labels[i])
        # plt.imshow(out, cmap='gray')
        # plt.show()



    X_train, X_test, Y_train, Y_test = train_test_split(np.asarray(X), np.asarray(y), test_size=0.2)
    Y_train = keras.utils.to_categorical(Y_train, len(label2char))
    Y_test = keras.utils.to_categorical(Y_test, len(label2char))
    json.dump(char2label, open('./data/katakana.char2label.json', 'w'))
    json.dump(label2char, open('./data/katakana.label2char.json', 'w'))

    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    pickle.dump((X_train, Y_train, X_test, Y_test), open('./data/katakana.pkl', 'wb'),  protocol=2)

#assume input 2d array
def preproces_binary_data(im):
    return im

def load_data_kata_quote():
    X_train, y_train, X_test, y_test = pickle.load(open('./data/full_katakana_quote.pkl', 'rb'))
    for i in range(X_train.shape[0]):
        X_train[i,0] = preproces_binary_data(X_train[i,0])
        # plt.imshow(X_train[i,0], cmap='gray')
        # plt.show()
    for i in range(X_test.shape[0]):
        X_test[i,0] = preproces_binary_data(X_test[i,0])
    # X_train=X_train[:500]
    # X_test = X_test[:500]
    # y_train = y_train[:500]
    # y_test = y_test[:500]
    pickle.dump((X_train, y_train, X_test, y_test), open('./data/full_katakana_quote2.pkl', 'wb'), protocol=2)

# load_data_kata_quote()
load_data_npz()