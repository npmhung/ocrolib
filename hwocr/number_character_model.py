import os
import sys
PATH_CUR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(PATH_CUR)
sys.path.append('../../../../')
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import random
import argparse
import numpy as np
import cv2
import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot as plt
from src.ocropy.ocrolib.hwocr import run_hw
import json
from scipy import misc
import keras
from keras.models import model_from_json
from keras import optimizers


def load_model_weights(name, model):
    try:
        model.load_weights(name)
    except Exception:
         print ("Can't load weights!")


def save_model_weights(name, model):
    try:
        model.save_weights(name)
    except Exception:
        print ("failed to save classifier weights")
        pass

def make_data():
    X_train, y_train, X_test, y_test, label2codes = pickle.load(open('./data/all.pkl', 'rb'))
    X_train = X_train[:60000]
    X_test = X_test[:10000]
    y_train = y_train[:60000]
    y_test = y_test[:10000]
    print(y_train.shape)
    print(X_train.shape)
    X_train2, y_train2, X_test2, y_test2 = pickle.load(open('./data/mnist_full.pkl', 'rb'))
    print(y_train2.shape)
    print(X_train2.shape)

    X_train_all = np.concatenate((X_train, X_train2), axis=0)
    X_test_all = np.concatenate((X_test, X_test2), axis=0)

    y_train_all = np.zeros(y_train.shape[0] + y_train2.shape[0])
    print(y_train_all.shape)
    y_train_all[:y_train.shape[0]] = 0
    y_train_all[y_train.shape[0]:] = 1

    y_test_all = np.zeros(y_test.shape[0] + y_test2.shape[0])
    print(y_test_all.shape)
    y_test_all[:y_test.shape[0]] = 0
    y_test_all[y_test.shape[0]:] = 1

    print(X_train_all.shape)
    print(y_train_all.shape)
    print(X_test_all.shape)
    print(y_test_all.shape)

    pickle.dump((X_train_all, y_train_all, X_test_all, y_test_all), open('./data/number_char.pkl', 'wb'), protocol=4)

def train(train_dir='./data/number_char.pkl'):
    from keras.callbacks import ModelCheckpoint
    from src.ocropy.ocrolib.hwocr.models import M7_1
    from keras import optimizers

    X_train, y_train, X_test, y_test = pickle.load(open(train_dir,'rb'))

    if len(y_train.shape)==1:
        y_train = keras.utils.to_categorical(y_train, 2)
        y_test= keras.utils.to_categorical(y_test, 2)

    n_output = 2
    model = M7_1(n_output=n_output, input_shape=(1, 64, 64))

    load_model_weights('./save/number_char_weights.h5', model)
    checkpoint = ModelCheckpoint('./save/number_char_weights.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    adam = optimizers.Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    model_json = model.to_json()
    with open("./save/model7_1_number_char.json", "w") as json_file:
        json_file.write(model_json)

    list_index = list(range(X_test.shape[0]))
    random.shuffle(list_index)

    print(y_test[list_index[:len(list_index)//2]][:10])

    model.fit(X_train, y_train, shuffle=True,
              epochs=100, callbacks= [checkpoint],
              batch_size=2048, validation_data=(X_test[list_index[:len(list_index)//2]],
                                        y_test[list_index[:len(list_index)//2]]))

    score, acc = model.evaluate(X_test[list_index[:len(list_index)//2]],
                                y_test[list_index[:len(list_index) // 2]],
                                batch_size=512,
                                verbose=1)
    print ("Training size: ", X_train.shape[0])
    print ("Test size: ", X_test.shape[0])
    print ("Test Score: ", score)
    print ("Test Accuracy: ", acc)


def predict(dir_img):
    from keras.models import model_from_json
    # print(label2codes)
    json_file = open("./save/model7_1_number_char.json", 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    load_model_weights('./save/number_char_weights.h5', model)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['accuracy'])
    new_images, im_names = run_hw.load_image_general(dir_img)

    for i in range(new_images.shape[0]):
        bimg=np.asarray(np.asarray([new_images[i]]))
        print(im_names[i])
        # print('probability ', probab_predict)
        result_predict = model.predict_classes(bimg, batch_size=1)

        print('predict {}'.format(result_predict))
        plt.imshow(new_images[i][0])
        plt.show()

def report2(root_dir):
    print(root_dir)
    from keras.models import model_from_json
    label2chars={"0":"not number", "1":"number"}
    json_file = open("./save/model7_1_number_char.json", 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    load_model_weights('./save/number_char_weights.h5', model)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['accuracy'])
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            if 'DAVID' not in name:
                filename = os.path.join(root, name)
                if filename[-4:]!='.txt' and \
                        ('.jpg' in filename or '.png' in filename or '.JPG' in filename or '.PNG' in filename):
                    str_re=run_hw.predict_single(filename, model, label2chars)
                    with open(filename[:-4]+'.txt','w', encoding='utf-8') as f:
                        f.write(str_re)

def eval():

    # print(label2codes)
    json_file = open("./save/model7_1_number_char.json", 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    load_model_weights('./save/number_char_weights.h5', model)
    X_train, y_train, X_test, y_test = pickle.load(open('./data/number_char.pkl', 'rb'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['accuracy'])
    # score, acc = model.evaluate(X_test[X_test.shape[0] // 2:], y_test[y_test.shape[0] // 2:],
    #                         batch_size=512,
    #                         verbose=1)
    # print(acc)
    list_index = list(range(X_test.shape[0]))
    random.shuffle(list_index)
    X_test=X_test[list_index]
    for i in range(X_test.shape[0]):
        bimg=np.asarray([X_test[i]])
        probab_predict = model.predict_proba(bimg, batch_size=1)
        top_index = np.argsort(probab_predict * -1)[0][0]
        result_predict = top_index
        # print('probability ', probab_predict)
        # result_predict = model.predict_classes(bimg, batch_size=1)
        print('predict {} '.format(result_predict))
        plt.imshow(X_test[i][0])
        plt.show()


if __name__=='__main__':
    # make_data()
    # train()
    # predict('./test_images/yamato_form1-20.2/word1/line4')
    # eval()
    report2('./test_images/yamato_form1-20.2')