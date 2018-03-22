import os
import sys
PATH_CUR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(PATH_CUR)
sys.path.append('../../../../')
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import shutil
import random
import argparse
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from ocrolib.table_cell_cut import cut_cell_region
import ocrolib.hwocr.label2code as label2code
import ocrolib.hwocr.run_hw as run_hw
from matplotlib import pyplot as plt
from keras.models import model_from_json
import cv2

label2codes = pickle.load(open('./data/label2codes.pkl', 'rb'))
# print(label2codes)
json_file = open("./save/model7_1.json", 'r')
loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
run_hw.load_model_weights('./save/M7_1-hiragana_weights.h5', model)

def predict_hw_single(dir_img='./tmp/hw_single_images/'):
    print('==================PREDICT SINGLE===================')

    new_images, im_names = run_hw.load_image(dir_img)
    final_str=''
    list_raw=[]
    for i in range(new_images.shape[0]):
        bimg=np.asarray([new_images[i]])
        fig = plt.figure()
        print(im_names[i])
        a = fig.add_subplot(1, 2, 1)
        a.set_title('Sample')
        rim=cv2.imread(im_names[i])
        plt.imshow(rim)
        list_raw.append(rim)
        probab_predict = model.predict_proba(bimg, batch_size=1)
        # print('probability ', probab_predict)
        result_predict = model.predict_classes(bimg, batch_size=1)
        jc=label2code.jis2unicode(label2codes[result_predict[0]])
        res=jc.encode('utf-8')
        print('predict {} -->{} '.format(result_predict,res ))
        final_str+=jc+' '
        a = fig.add_subplot(1, 2, 2)
        a.set_title('Binary')
        plt.imshow(new_images[i][0])
        plt.show()
    return final_str, list_raw


def predict_hw_full(dir_img, tmp_dir='./tmp/hw_single_images/'):
    try:
        shutil.rmtree(tmp_dir)
    except:
        pass
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
    cut_cell_region.regionCutCell(dir_img, os.path.basename(dir_img), tmp_dir)
    return predict_hw_single(tmp_dir)

def predict_ocr(dir_img, tmp_dir='./tmp/'):
    img = cv2.imread(dir_img, cv2.CV_8UC1)
    out = cv2.blur(img, (3, 3))
    out = cv2.adaptiveThreshold(out, 1.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    out = cv2.medianBlur(out, 5)
    # plt.imshow(out)
    # plt.show()
    tmp_dirimg=tmp_dir+'./timg.png'
    cv2.imwrite(tmp_dirimg, np.asarray(out*256, np.int32))
    os.system("tesseract {} {}/out -l jpn+jpn_vert --oem 0 --psm 6".format(tmp_dirimg,tmp_dir))
    with open('{}/out.txt'.format(tmp_dir),'r') as f:
        str=''
        for l in f:
            str+=l
        return str


def report(root_dir):
    print(root_dir)
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            if 'DAVID' not in name:
                filename = os.path.join(root, name)
                if filename[-4:]!='.txt' and 'line' in filename:
                    str_re, list_raw=predict_hw_full(filename)
                    with open(filename[:-4]+'.txt','w', encoding='utf-8') as f:
                        f.write(str_re)
                    single_folder=filename[:-4]
                    if not os.path.isdir(single_folder):
                        os.mkdir(single_folder)
                    for ind, rim in enumerate(list_raw):
                        cv2.imwrite(single_folder+'/DAVID'+str(ind)+'.png', rim)


def predict(dir_img, mode_text='hw'):
    res = ''
    if mode_text=='hw':
        res=predict_hw_full(dir_img)
    elif mode_text=='type':
        res=predict_ocr(dir_img)

    return res

if __name__=='__main__':
    # predict_hw_full('./test_images/Yamato_input/form1/line/4.jpg')
    # print(predict_ocr('./test_images/Yamato_input/form1/line/1.jpg'))
    report('./test_images/run_4_templates/')