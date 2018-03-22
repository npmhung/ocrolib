import struct
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from skimage.filters import threshold_otsu
from sklearn.model_selection import train_test_split
import keras
import cv2
import json
import pickle
sz_record = 576


def load_class2JIStable(filepath):
    f = open(filepath, 'r')
    table = {}
    for i, line in enumerate(f):
        try:
            fields = line.strip().split(' ')
            table[int(fields[0])] = fields[-1].strip()[-4:]
        except ValueError:
            print("Read table error at line ", i)
            pass

    f.close()
    return table

def jis2unicode(jis_code):
    b = b'\033$B' + bytes.fromhex(jis_code)
    c = b.decode('iso2022_jp')
    return c


def read_record_ETL9B(f):
    s = f.read(sz_record)
    r = struct.unpack('>2H4s504s64x', s)
    iF = Image.frombytes('1', (64, 63), r[3], 'raw')

    return r + (iF,)


def add_trasform_to_bin_img(img, bor=0, num_erode=0):
    # print('{} vs {}'.format(bor, num_erode))
    img = cv2.copyMakeBorder(img, top=bor, bottom=bor, left=bor, right=bor,
                                 borderType=cv2.BORDER_CONSTANT, value=[1])
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)

    kernel = np.ones((2, 2), np.uint8)
    if num_erode>0:
        img = 1 - cv2.erode(1 - img, kernel, iterations=num_erode)
    return img

def read_etl9():
    # Character type = 72, person = 160, y = 127, x = 128    
    ary = np.zeros([3036*200, 1, 64, 64], dtype=np.uint8)
    c=0
    clabels=[]
    label2char={}
    char2label={}
    table = load_class2JIStable('./data/etl9b.txt')
    for j in range(1, 6):
        filename = './data/ETL9B/ETL9B_{:01d}'.format(j)
        print("Reading from data file ", filename)
        with open(filename, 'rb') as f:
            # skip first dummy record
            skip = 1
            record_size = 576
            f.seek(skip * record_size)            

            for id_dataset in range(40):
                moji = 0
                for i in range(3036):
                    r = read_record_ETL9B(f)
                    cim = np.array(r[-1]) * 1
                    ary[c,0,:63,:] =  cim
                    temp = 1.0-ary[c,0]
                    # if np.random.rand()>0.3:
                    #     border=np.random.choice([3, 5, 7])
                    #     iter = np.random.choice([1])
                    #     temp = add_trasform_to_bin_img(temp, border, iter)
                    ary[c, 0] = temp
                    # if np.random.rand()>0.8:
                    #     plt.imshow(temp, cmap='gray')
                    #     plt.show()
                    ary[c, 0] = temp


                    ch=jis2unicode(table[i])
                    if ch not in char2label:
                        char2label[ch]=i
                        label2char[str(i)]=ch
                    clabels.append(i)
                    c+=1
                    # plt.imshow(cim, cmap='gray')
                    # plt.show()
                    moji += 1
    print(ary.shape)

    for t in range(10):
        i = int(np.random.choice(ary.shape[0], 1))
        print(label2char[str(clabels[i])])
     #   plt.imshow(ary[i, 0, :, :], cmap='gray')
     #   plt.show()

    X_train, X_test, Y_train, Y_test = train_test_split(np.asarray(ary), np.asarray(clabels), test_size=0.2)
    # Y_train = keras.utils.to_categorical(Y_train, len(label2char))
    # Y_test = keras.utils.to_categorical(Y_test, len(label2char))
    json.dump(char2label, open('./data/kanji_hira_plain.char2label.json', 'w'))
    json.dump(label2char, open('./data/kanji_hira_plain.label2char.json', 'w'))

    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    pickle.dump((X_train, Y_train, X_test, Y_test), open('./data/kanji_hira_plain.pkl', 'wb'),protocol=4)


read_etl9()

